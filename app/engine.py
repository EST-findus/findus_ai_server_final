"""모델 실행. ai_lab 의 /final 화면(web/server.py final_run, _final_hair)과 같은 순서·설정으로 돈다.

GPU 하나(6GB)라서
  - 한 번에 한 요청만 GPU 를 쓴다 (나머지는 QUEUE_TIMEOUT 초까지 줄 서서 기다린다).
  - 나이 변환(SAM)과 머리(HairFastGAN 5.5GB)는 같이 올라가지 않는다. 나이 변환 전에 머리 모델을 내리고,
    머리 모델은 스타일만 바꿔 다시 요청할 때를 위해 다음 나이 변환 전까지 올려 둔다 (다시 올리면 약 17초).
  - 나이 변환은 configs/final.yaml 의 keep_loaded(기본 false) 를 따른다. false 면 단계마다 올렸다 내린다.
"""
import gc
import pickle
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
from src.paths import WEIGHTS
from src.pipeline import Job, Pipeline, _Work, imread_rgb, imwrite_rgb

from .settings import CONFIG, ENCODER_DIR, HAIR_PRESETS, HAIR_REFS, QUEUE_TIMEOUT, SAM_BASE

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


def default_encoder() -> str:
    """configs/final.yaml 의 나이 인코더 파일 이름 (encoder 를 안 고르면 이것)."""
    return Path(config()["aging"]["encoder_full"]).name


def encoders() -> List[str]:
    """weights/sam/ 에 넣은 나이 인코더 파일 이름. SAM 본체와 기본 인코더(화면에 '(기본)'으로 따로 나옴)는 뺀다."""
    skip = {SAM_BASE, default_encoder()}
    return sorted(p.name for p in ENCODER_DIR.glob("*.pt") if p.name not in skip) if ENCODER_DIR.is_dir() else []


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
        self._encoder = None      # _cache 에 올라간 모델이 쓴 나이 인코더 경로
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

    # ---------------------------------------------------------------- 0. 복원 (골랐을 때만, 나이 변환 전에)

    def restore(self, img: np.ndarray, w: float = 0.5) -> tuple:
        """사진 → {'face': 복원 얼굴, 'input': 복원 전 정렬 얼굴} (FFHQ 정렬 1024), 정보.
        흐름: 검출 → FFHQ 정렬 → 흑백이면 컬러화 → CodeFormer 복원 → 원본 대비 ArcFace.
        w: CodeFormer 충실도 0~1. 0 에 가까울수록 강하게 보정(화질 우선), 1 에 가까울수록 원본 유지.
        final.yaml 은 복원을 꺼 두었다 (신원을 깎았다, ai_lab docs/[15]). 그래서 결과를 눈으로 보고 고르게 따로 둔다."""
        if not (WEIGHTS / "codeformer" / "codeformer.pth").exists():
            raise BadInput("복원 가중치가 없습니다: weights/codeformer/codeformer.pth (CodeFormer v0.1.0 릴리스에서 받기)")
        cfg = config()
        cfg["keep_loaded"] = False  # 가끔 쓰는 단계라 쓰고 바로 내린다
        cfg["restorer"] = {"name": "codeformer", "position": "before", "w": float(w)}
        cfg["evaluators"] = [e for e in cfg["evaluators"] if e["name"] == "arcface"]
        with self.gpu():
            self._free_hair()
            self._free_aging()
            pipe = Pipeline(cfg)
            works = [_Work(Job(img), [])]
            pipe._section(pipe._detect_all, works)
            if pipe.colorize_on:
                pipe._section(pipe._colorize_all, works)
            pipe._section(pipe._restore_all, works, "before")
            work = works[0]
            if work.error is None:
                work.result.outputs = {0: work.result.restored}  # 평가 섹션은 outputs 를 원본 사진과 비교한다
                pipe._section(pipe._evaluate_all, works)
        if isinstance(work.error, ValueError):  # 얼굴을 못 찾음
            raise BadInput(str(work.error))
        if work.error is not None:
            raise work.error
        r = work.result
        images = {"face": r.restored, "input": r.gray_input if r.colorized else r.aligned}
        return images, {"w": float(w), "colorized": bool(r.colorized),
                        "color_spread": None if r.color_spread is None else round(r.color_spread, 2),
                        "identity": {"face": {n: round(v["vs_input"], 3) for n, v in r.scores[0].items()}}}

    # ---------------------------------------------------------------- 1. 나이 변환

    def age(self, img: np.ndarray, ages: List[int], sex: str, gender_strength: float, strength: float = 1.0,
            encoder: Optional[str] = None, original: Optional[np.ndarray] = None) -> tuple:
        """아이 사진 → {'age30': 얼굴, ...} (FFHQ 정렬 1024), 정보.
        흐름: 검출(RetinaFace) → FFHQ 정렬 → 흑백이면 컬러화(CodeFormer) → SAM(나이 인코더) + 성별 보정.
        encoder: weights/sam/ 의 파일 이름. 비우면 configs/final.yaml 의 기본(best_encoder.pt).
        original: img 가 복원 얼굴일 때 복원 전 얼굴. 주면 원본 대비 신원 점수(identity_original)도 낸다."""
        cfg = config()
        cfg["aging"]["params"].update(sex=sex, strength=float(strength), gender_strength=float(gender_strength))
        cfg["evaluators"] = [e for e in cfg["evaluators"] if e["name"] == "arcface"]  # AdaFace 는 서버에 코드·가중치가 없다
        if encoder:
            if encoder not in encoders():  # 목록에 있는 이름만 (임의 경로 접근 막기)
                raise BadInput(f"모르는 나이 인코더입니다: {encoder} (GET /v1/aging/encoders)")
            cfg["aging"]["encoder_full"] = str(ENCODER_DIR / encoder)
        with self.gpu():
            self._free_hair()
            if cfg["aging"]["encoder_full"] != self._encoder:  # keep_loaded 일 때 이전 인코더가 GPU 에 남지 않게
                self._free_aging()
                self._encoder = cfg["aging"]["encoder_full"]
            try:
                res = Pipeline(cfg, cache=self._cache).run(img, target_ages=ages, reference=original)
            except ValueError as e:  # 얼굴을 못 찾음
                raise BadInput(str(e))
            except (KeyError, TypeError, RuntimeError, pickle.UnpicklingError) as e:
                if encoder and (not isinstance(e, RuntimeError) or "state_dict" in str(e)):
                    raise BadInput(f"나이 인코더 형식이 아닙니다: {encoder} ({type(e).__name__}). "
                                   "docs/[03] 2-2 의 형식인지 확인하세요.") from e
                raise
        images = {f"age{a}": out for a, out in sorted(res.outputs.items())}
        info = {"encoder": Path(cfg["aging"]["encoder_full"]).name,
                "colorized": bool(res.colorized),
                "color_spread": None if res.color_spread is None else round(res.color_spread, 2),
                # 입력 사진 대비 신원 점수 (코사인). 기록만 하고 걸러내지 않는다: 기준값은 dev 평가 뒤에 정한다
                "identity": {f"age{a}": {n: round(v["vs_input"], 3) for n, v in s.items()}
                             for a, s in sorted(res.scores.items())}}
        if original is not None:  # 복원 얼굴로 변환했으면 복원 전 원본 대비 점수도
            info["identity_original"] = {f"age{a}": {n: round(v["vs_reference"], 3) for n, v in s.items()}
                                         for a, s in sorted(res.scores.items())}
        return images, info

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
