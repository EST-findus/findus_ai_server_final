"""과거 사진 → (검출 → 정렬 → 복원 → 나이 변환) → 미래 얼굴 + 평가 점수.

    pipe = Pipeline.from_yaml("configs/default.yaml")
    result = pipe.run(img_rgb, target_ages=[20, 30], reference=future_rgb)
    results = pipe.run_many([Job(img1), Job(img2, target_ages=[40])])   # 여러 장은 섹션별로 몰아서

GPU 메모리가 작아서(6GB) 모델을 한꺼번에 올리지 않고 섹션 단위로 실행한다.
섹션마다 필요한 모델만 올려 모든 사진을 처리한 뒤 내리고 다음 섹션으로 넘어간다.
    검출(+사전 업스케일) → 흑백 컬러화 → 복원(before) → 나이 변환 → 복원(after) → 후 업스케일 → 평가(검출 + 평가모델)
keep_loaded: true 면 한 번 올린 모델을 계속 둔다 (VRAM 여유가 있을 때 속도용).
"""
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import yaml

from .align import arcface_align, ffhq_align
from .registry import build
from .stages.colorize import color_spread
from .stages.evaluate import cosine

NONE = {"name": "none"}


@dataclass
class Result:
    aligned: np.ndarray                                   # FFHQ 정렬 원본 (1024)
    restored: np.ndarray                                  # 복원 후 (나이 변환 입력)
    outputs: Dict[int, np.ndarray] = field(default_factory=dict)              # 나이 → 결과 이미지
    scores: Dict[int, Dict[str, Dict[str, float]]] = field(default_factory=dict)  # 나이 → 평가모델 → 점수
    baseline: Dict[str, float] = field(default_factory=dict)  # 평가모델 → 원본 어릴 때 사진 ↔ 실제 미래 사진
    recon: Optional[np.ndarray] = None                    # aging.show_recon: 나이 변환 없이 재구성만 한 결과
    recon_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    color_spread: Optional[float] = None                  # 정렬 얼굴의 색 퍼짐 (작을수록 흑백, colorizer 켰을 때만)
    colorized: bool = False                               # 흑백으로 판단해 컬러화했는가
    gray_input: Optional[np.ndarray] = None               # 컬러화 전 정렬 얼굴 (colorized 일 때만)


@dataclass
class Job:
    """lm5 / reference_lm5 를 주면 검출을 건너뛰고 그 랜드마크(예: AI Hub 라벨)를 쓴다."""
    image: np.ndarray
    target_ages: Optional[List[int]] = None
    reference: Optional[np.ndarray] = None
    lm5: Optional[np.ndarray] = None
    reference_lm5: Optional[np.ndarray] = None


@dataclass
class _Work:
    """섹션 사이에 넘기는 사진 한 장의 중간 결과. 이미지는 전부 CPU(numpy)에 둔다."""
    job: Job
    ages: List[int]
    img: Optional[np.ndarray] = None          # 검출에 쓴 이미지 (사전 업스케일됐을 수 있음)
    lm5: Optional[np.ndarray] = None
    ref_img: Optional[np.ndarray] = None
    ref_lm5: Optional[np.ndarray] = None
    aligned_lm5: Optional[np.ndarray] = None
    result: Optional[Result] = None
    error: Optional[Exception] = None


def _detach(e: Exception) -> Exception:
    """예외의 traceback 이 섹션 함수의 프레임(=모델 참조)을 붙잡아 GPU 메모리가 안 풀리는 것을 막는다."""
    x = e
    while x is not None:
        x.__traceback__ = None
        x = x.__cause__ or x.__context__
    return e


class Pipeline:
    def __init__(self, cfg: dict, cache: Optional[dict] = None):
        """cache 는 keep_loaded 일 때만 쓰인다 (여러 Pipeline 이 이미 올린 모델을 공유, 웹 UI용)."""
        self.cfg = cfg
        self.device = cfg.get("device", "cuda")
        self.keep_loaded = cfg.get("keep_loaded", False)
        self._cache = {} if cache is None else cache
        self.evaluator_names = [e["name"] for e in cfg.get("evaluators", [])]

    @classmethod
    def from_yaml(cls, path, **overrides):
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        cfg.update(overrides)
        return cls(cfg)

    # ---------------------------------------------------------------- 모델 올리기/내리기

    def _build(self, stage: str, spec: dict):
        kwargs = {k: v for k, v in spec.items()
                  if k not in ("name", "position", "min_face", "target_ages", "max_side", "params", "show_recon")}
        if not self.keep_loaded:
            return build(stage, spec["name"], device=self.device, **kwargs)
        key = (stage, spec["name"], tuple(sorted(kwargs.items())))
        if key not in self._cache:
            self._cache[key] = build(stage, spec["name"], device=self.device, **kwargs)
        return self._cache[key]

    def _free(self):
        if self.keep_loaded:
            return
        self._cache.clear()
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _section(self, fn, works: List[_Work], *args):
        """섹션 하나 실행. 섹션 안에서 만든 모델은 함수 지역변수라 끝나면 참조가 사라지고, 여기서 GPU 캐시까지 비운다."""
        try:
            fn(works, *args)
        finally:
            self._free()

    @staticmethod
    def _each(works: List[_Work], fn):
        """실패한 사진은 건너뛰고, 새로 실패하면 기록만 하고 나머지는 계속 처리한다."""
        for w in works:
            if w.error is not None:
                continue
            try:
                fn(w)
            except Exception as e:
                w.error = _detach(e)

    @classmethod
    def _map_outputs(cls, works: List[_Work], fn):
        def work(w):
            r = w.result
            r.outputs = {age: fn(out) for age, out in r.outputs.items()}
            if r.recon is not None:
                r.recon = fn(r.recon)
        cls._each(works, work)

    # ---------------------------------------------------------------- 검출 도우미

    def _detect(self, detector, img_rgb: np.ndarray):
        """긴 변이 max_side 보다 크면 줄여서 검출하고 좌표를 원래 크기로 되돌린다.
        (얼굴이 사진을 꽉 채우는 고해상도 사진에서 RetinaFace가 눈·입을 각각 얼굴로 잡는 문제 방지)"""
        max_side = self.cfg["detector"].get("max_side", 1024)
        s = max_side / max(img_rgb.shape[:2])
        if s >= 1:
            return detector(img_rgb)
        small = cv2.resize(img_rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        faces = detector(small)
        for f in faces:
            f.bbox = np.asarray(f.bbox, np.float32) / s
            f.landmarks5 = f.landmarks5 / s
            if f.landmarks68 is not None:
                f.landmarks68 = f.landmarks68 / s
        return faces

    def _landmarks_or(self, detector, img_rgb: np.ndarray, fallback_lm5: np.ndarray) -> np.ndarray:
        faces = self._detect(detector, img_rgb)
        return faces[0].landmarks5 if faces else fallback_lm5

    # ---------------------------------------------------------------- 섹션

    def _detect_all(self, works: List[_Work]):
        detector = self._build("detector", self.cfg["detector"])
        pre = self.cfg.get("pre_upscale", NONE)
        upscaler = None  # 작은 얼굴이 나왔을 때만 올린다

        def detect(img_rgb):
            """가장 큰 얼굴 하나. 얼굴이 min_face 보다 작으면 업스케일 후 다시 검출한다."""
            nonlocal upscaler
            faces = self._detect(detector, img_rgb)
            too_small = not faces or (faces[0].bbox[2] - faces[0].bbox[0]) < pre.get("min_face", 0)
            if too_small and pre["name"] != "none":
                upscaler = upscaler or self._build("upscaler", pre)
                if getattr(upscaler, "scale", 1) > 1:
                    img_rgb = upscaler(img_rgb)
                    faces = self._detect(detector, img_rgb)
            if not faces:
                raise ValueError("얼굴을 찾지 못했습니다.")
            return img_rgb, faces[0].landmarks5

        def work(w):
            j = w.job
            if j.lm5 is not None:
                w.img, w.lm5 = j.image, np.asarray(j.lm5, np.float32)
            else:
                w.img, w.lm5 = detect(j.image)
            aligned = ffhq_align(w.img, w.lm5, output_size=1024)
            w.result = Result(aligned=aligned, restored=aligned)
            if not self.evaluator_names:
                return
            # 평가 섹션에서 쓸 랜드마크를 검출기가 올라와 있을 때 미리 구한다
            w.aligned_lm5 = self._landmarks_or(detector, aligned, w.lm5)
            if j.reference is not None:
                if j.reference_lm5 is not None:
                    w.ref_img, w.ref_lm5 = j.reference, np.asarray(j.reference_lm5, np.float32)
                else:
                    w.ref_img, w.ref_lm5 = detect(j.reference)

        self._each(works, work)

    @property
    def colorize_on(self) -> bool:
        spec = self.cfg.get("colorizer", NONE)
        return spec["name"] != "none" and spec.get("when", "auto") != "never"

    def _colorize_all(self, works: list, faces=lambda w: [w.result]):
        """흑백(세피아 포함)으로 보이는 사진만 컬러화한다. 판단은 나이 변환에 실제로 들어가는 정렬 얼굴로 한다.
        faces: 작업 하나에서 컬러화 대상(aligned/restored 를 가진 객체)을 꺼내는 함수 (부모 반영은 아이 + 부모).
        컬러 사진을 흑백으로 잘못 보면 원래 색을 버리게 되므로 임계값은 보수적으로 둔다 (docs/[08])."""
        spec = self.cfg["colorizer"]
        when = spec.get("when", "auto")
        threshold = spec.get("gray_threshold", 3.0)
        todo = {}

        def check(w):
            for f in faces(w):
                f.color_spread = color_spread(f.aligned)
                if when == "always" or f.color_spread < threshold:
                    todo.setdefault(id(w), []).append(f)

        self._each(works, check)
        if not todo:
            return
        # 흑백 사진은 가끔만 들어오므로 keep_loaded 여도 캐시에 두지 않고 쓰고 바로 내린다 (6GB)
        kwargs = {k: v for k, v in spec.items() if k not in ("name", "when", "gray_threshold")}
        colorizer = build("colorizer", spec["name"], device=self.device, **kwargs)

        def work(w):
            for f in todo.get(id(w), []):
                f.gray_input = f.aligned
                f.aligned = f.restored = colorizer(f.aligned)
                f.colorized = True

        try:
            self._each(works, work)
        finally:
            del colorizer
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _restore_all(self, works: List[_Work], when: str):
        restorer = self._build("restorer", self.cfg.get("restorer", NONE))
        if when == "before":
            self._each(works, lambda w: setattr(w.result, "restored", restorer(w.result.aligned)))
        else:
            self._map_outputs(works, restorer)

    def _age_all(self, works: List[_Work]):
        aging = self._build("aging", self.cfg["aging"])
        # aging.params 는 모델 생성자가 아니라 호출 시점에 넘긴다 (바꿔도 모델을 다시 로드하지 않음)
        params = self.cfg["aging"].get("params", {})
        recon = self.cfg["aging"].get("show_recon") and hasattr(aging, "reconstruct")

        def work(w):
            r = w.result
            if recon:
                r.recon = aging.reconstruct(r.restored, **params)
            for age in w.ages:
                r.outputs[age] = aging(r.restored, age, **params)

        self._each(works, work)

    def _post_upscale_all(self, works: List[_Work]):
        self._map_outputs(works, self._build("upscaler", self.cfg.get("post_upscale", NONE)))

    def _evaluate_all(self, works: List[_Work]):
        detector = self._build("detector", self.cfg["detector"])
        evaluators = {e["name"]: self._build("evaluator", e) for e in self.cfg.get("evaluators", [])}

        def embed(img_rgb, lm5):
            face112 = arcface_align(img_rgb, lm5)
            return {name: ev.embed(face112) for name, ev in evaluators.items()}

        def work(w):
            r = w.result
            src_emb = embed(w.img, w.lm5)
            ref_emb = embed(w.ref_img, w.ref_lm5) if w.ref_img is not None else None
            if ref_emb:
                r.baseline = {name: cosine(src_emb[name], ref_emb[name]) for name in ref_emb}

            def score(out):
                scale = out.shape[0] / r.aligned.shape[0]
                out_emb = embed(out, self._landmarks_or(detector, out, w.aligned_lm5 * scale))
                return {name: {"vs_input": cosine(src_emb[name], e),
                               **({"vs_reference": cosine(ref_emb[name], e)} if ref_emb else {})}
                        for name, e in out_emb.items()}

            if r.recon is not None:
                r.recon_scores = score(r.recon)
            r.scores = {age: score(out) for age, out in r.outputs.items()}

        self._each(works, work)

    # ---------------------------------------------------------------- 실행

    def run_many(self, jobs: List[Job]) -> List[Union[Result, Exception]]:
        """사진 여러 장을 섹션별로 몰아서 처리한다. 실패한 사진은 Result 대신 예외가 들어간다."""
        default_ages = self.cfg["aging"].get("target_ages", [30])
        works = [_Work(j, [int(a) for a in (j.target_ages or default_ages)]) for j in jobs]
        position = self.cfg.get("restorer", {}).get("position", "before")

        self._section(self._detect_all, works)
        if self.colorize_on:
            self._section(self._colorize_all, works)
        if position in ("before", "both"):
            self._section(self._restore_all, works, "before")
        self._section(self._age_all, works)
        if position in ("after", "both"):
            self._section(self._restore_all, works, "after")
        self._section(self._post_upscale_all, works)
        if self.evaluator_names:
            self._section(self._evaluate_all, works)
        return [w.error or w.result for w in works]

    def run(self, img_rgb: np.ndarray, target_ages: Optional[List[int]] = None,
            reference: Optional[np.ndarray] = None, lm5: Optional[np.ndarray] = None,
            reference_lm5: Optional[np.ndarray] = None) -> Result:
        """lm5 / reference_lm5 를 주면 검출을 건너뛰고 그 랜드마크(예: AI Hub 라벨)를 쓴다."""
        out = self.run_many([Job(img_rgb, target_ages, reference, lm5, reference_lm5)])[0]
        if isinstance(out, Exception):
            raise out
        return out


def imread_rgb(path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)  # 한글 경로 대응
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path, img_rgb: np.ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(path)
    buf.tofile(str(path))
