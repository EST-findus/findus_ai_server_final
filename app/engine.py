"""모델 실행. ai_lab 의 /final 화면(web/server.py final_run, _final_hair)과 같은 순서·설정으로 돈다.

GPU 하나(6GB)라서
  - 한 번에 한 요청만 GPU 를 쓴다 (나머지는 QUEUE_TIMEOUT 초까지 줄 서서 기다린다).
  - 나이 변환(SAM)과 머리(HairFastGAN 5.5GB)는 같이 올라가지 않는다. 나이 변환 전에 머리 모델을 내리고,
    머리 모델은 스타일만 바꿔 다시 요청할 때를 위해 다음 나이 변환 전까지 올려 둔다 (다시 올리면 약 17초).
  - 나이 변환은 configs/final.yaml 의 keep_loaded(기본 false) 를 따른다. false 면 단계마다 올렸다 내린다.
"""
import gc
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml

from src.align import ffhq_align
from src.hair_color import HairMask, hair_delta_e, recolor
from src.hairstyle import HairPost, hair_only
from src.pipeline import Pipeline, imread_rgb, imwrite_rgb

from .settings import CONFIG, HAIR_PRESETS, HAIR_REFS, QUEUE_TIMEOUT

HAIR_ALIGNED = HAIR_REFS / "_aligned"    # 참조 사진 FFHQ 정렬 캐시
HAIR_THUMBS = HAIR_REFS / "_hair_only"   # 목록 화면용 썸네일 (참조 속 얼굴은 지우고 머리만)


class Busy(Exception):
    """GPU 를 QUEUE_TIMEOUT 초 안에 얻지 못함."""


class BadInput(Exception):
    """요청 값이 잘못됨 (400)."""


class NotReady(Exception):
    """아직 구현하지 않은 기능 (501)."""


def config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def presets() -> dict:
    return yaml.safe_load(HAIR_PRESETS.read_text(encoding="utf-8"))


def style_refs(sex: str, key: str) -> List[str]:
    """스타일 참조 사진 이름. selected_refs(품질 선정본)가 있으면 그 사진만, 없으면 폴더에서 exclude_refs 를 뺀다 (ai_lab docs/[20])."""
    d = HAIR_REFS / sex / key
    if sex not in ("M", "F") or not d.is_dir():
        return []
    pr = presets()
    have = sorted(q.name for q in d.glob("*") if q.suffix.lower() in (".png", ".jpg", ".jpeg"))
    chosen = (pr.get("selected_refs") or {}).get(f"{sex}/{key}")
    if chosen:
        return [n for n in chosen if n in have]
    skip = set(pr.get("exclude_refs") or [])
    return [n for n in have if f"{sex}/{key}/{n}" not in skip]


def resolve_color(color: Optional[str]) -> Optional[str]:
    """색 key('ash_brown') 또는 '#RRGGBB' → '#RRGGBB'."""
    if not color:
        return None
    for c in presets()["colors"]:
        if color == c["key"]:
            return c["hex"]
    import re
    if re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        return color
    raise BadInput(f"모르는 머리 색입니다: {color} (GET /v1/hair/styles 의 colors key 또는 #RRGGBB)")


def _empty_cuda():
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Engine:
    def __init__(self):
        self._lock = threading.Lock()
        self._cache = {}          # keep_loaded: true 일 때 나이 변환 모델 보관
        self._hair: Optional[HairPost] = None
        self._thumb_lock = threading.Lock()
        self._thumb_parser = None

    # ---------------------------------------------------------------- GPU 차례

    @contextmanager
    def gpu(self):
        if not self._lock.acquire(timeout=QUEUE_TIMEOUT):
            raise Busy(f"다른 요청이 GPU 를 쓰고 있습니다 ({QUEUE_TIMEOUT:.0f}초 대기 초과). 잠시 뒤 다시 시도하세요.")
        try:
            yield
        finally:
            self._lock.release()

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    @property
    def loaded(self) -> List[str]:
        return (["hair"] if self._hair is not None and self._hair.styler is not None else []) + \
               (["aging"] if self._cache else [])

    def _free_hair(self):
        if self._hair is not None:
            self._hair.free()
            self._hair = None
            _empty_cuda()

    def _free_aging(self):
        if self._cache:
            self._cache.clear()
            _empty_cuda()

    # ---------------------------------------------------------------- 1. 나이 변환

    def age(self, img: np.ndarray, ages: List[int], sex: str, gender_strength: float, strength: float = 1.0) -> tuple:
        """아이 사진 → {'age30': 얼굴, ...} (FFHQ 정렬 1024), 정보.
        흐름: 검출(RetinaFace) → FFHQ 정렬 → 흑백이면 컬러화(CodeFormer) → SAM(나이 인코더 best_encoder.pt) + 성별 보정."""
        cfg = config()
        cfg["aging"]["params"].update(sex=sex, strength=float(strength), gender_strength=float(gender_strength))
        cfg["evaluators"] = []  # 서버에서는 채점하지 않는다 (실제 성인 사진이 없음)
        with self.gpu():
            self._free_hair()
            try:
                res = Pipeline(cfg, cache=self._cache).run(img, target_ages=ages)
            except ValueError as e:  # 얼굴을 못 찾음
                raise BadInput(str(e))
        images = {f"age{a}": out for a, out in sorted(res.outputs.items())}
        return images, {"colorized": bool(res.colorized),
                        "color_spread": None if res.color_spread is None else round(res.color_spread, 2)}

    # ---------------------------------------------------------------- 2. 머리 스타일·색

    def _aligned_ref(self, pipe: Pipeline, detector, sex: str, key: str, name: str) -> np.ndarray:
        cached = HAIR_ALIGNED / sex / key / (Path(name).stem + ".png")
        if cached.exists():
            return imread_rgb(cached)
        img = imread_rgb(HAIR_REFS / sex / key / name)
        faces = pipe._detect(detector, img)
        if not faces:
            raise BadInput(f"참조 사진에서 얼굴을 찾지 못했습니다: {sex}/{key}/{name}")
        out = ffhq_align(img, faces[0].landmarks5, output_size=1024)
        imwrite_rgb(cached, out)
        return out

    def hair(self, images: Dict[str, np.ndarray], sex: str, style: Optional[str], ref: str,
             color_hex: Optional[str], scale: float, align: bool) -> tuple:
        """{키: 얼굴} → {키: 머리 바꾼 얼굴}, {키: 정보}. align=True 면 올린 사진이라 먼저 검출·정렬한다.
        모양은 HairFastGAN (원래 머리 지우기 → 참조 배경 지우기 → 얼굴 폭 비슷한 참조 고르기 → 얼굴 지키기 합성),
        색은 머리 영역 Lab 칠하기 (src/hair_color.py)."""
        cfg = config()
        names = []
        if style:
            names = style_refs(sex, style) if ref == "auto" else [ref]
            if not names or (ref != "auto" and ref not in style_refs(sex, style)):
                raise BadInput(f"머리 스타일 참조가 없습니다: {sex}/{style}/{ref}")
        with self.gpu():
            self._free_aging()
            if self._hair is None:
                self._hair = HairPost(detector=cfg["detector"])
            post = self._hair
            if align:
                try:
                    images = post.align(images)
                except ValueError as e:
                    raise BadInput(str(e))
            out = dict(images)
            info = {k: {} for k in images}
            if style:
                todo = [n for n in names if not (HAIR_ALIGNED / sex / style / (Path(n).stem + ".png")).exists()]
                pipe, detector = post.pipe, None
                if todo:
                    detector = pipe._build("detector", cfg["detector"])
                try:
                    refs = [self._aligned_ref(pipe, detector, sex, style, n) for n in names]
                finally:
                    del detector
                    pipe._free()
                res = post.run_many(out, (refs[0], False), candidates=[(r, False) for r in refs[1:]],
                                    fit=False, scale=scale)
                out = {k: img for k, (img, _) in res.items()}
                for k, f in post.last_fit.items():
                    info[k]["ref"] = names[f["ref"]]
                    info[k]["scale"] = f["scale"]
            if color_hex:
                hm = HairMask()
                try:
                    for k, img in out.items():
                        m = hm(img)
                        out[k] = recolor(img, m, color_hex)
                        info[k]["color_dE"] = round(hair_delta_e(out[k], m, color_hex), 2)
                finally:
                    hm.free()
        return out, info

    def thumbnail(self, sex: str, key: str, name: str) -> Path:
        """참조 사진 → 머리만 남긴 썸네일 (512, 처음 한 번 만들어 둔다). 얼굴 분할은 CPU 로 (GPU 작업과 안 겹치게)."""
        out = HAIR_THUMBS / sex / key / (Path(name).stem + ".jpg")
        if not out.exists():
            with self._thumb_lock:
                if self._thumb_parser is None:
                    self._thumb_parser = HairMask("cpu")
                img = cv2.resize(imread_rgb(HAIR_REFS / sex / key / name), (512, 512), interpolation=cv2.INTER_AREA)
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), cv2.cvtColor(hair_only(img, self._thumb_parser), cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
        return out

    # ---------------------------------------------------------------- 3. 안경 (아직 없음)

    def glasses(self, images: Dict[str, np.ndarray], style: str, align: bool) -> tuple:
        raise NotReady("안경 씌우기는 아직 구현 전입니다. 요청 형식은 docs/[00]_서버_구성_계획.md 5-3 을 보세요.")


engine = Engine()
