"""안경 씌우기: 투명 배경 안경 사진(PNG)을 FFHQ 정렬 얼굴의 두 눈에 맞춰 합성한다 (docs/[04]).

새 모델 없이 얼굴 분할(BiSeNet 19부위, src/hair_color.HairMask.parse) 하나만 쓴다.
  1) 두 눈(라벨 4·5) 중심 → 안경 두 렌즈 중심을 겹치게 크기·각도를 맞춘다.
  2) 테 그림자를 피부·코(1·10)에만 옅게, 렌즈에는 아주 옅은 반사만.
  3) 얼굴 옆으로 나가는 테 끝은 머리카락(17)이 덮으면 지운다.
  4) 원본 유지: 테·그림자·렌즈가 닿지 않은 픽셀은 입력을 그대로 복사한다 (계산 오차로도 안 바뀐다).

에셋은 scripts/make_glasses_assets.py 가 data/glasses/ 에 만든다: <key>.png(RGBA), <key>_lens.png, <key>.yaml.

    parse = HairMask("cpu").parse(face)
    out, info = apply(face, parse, load("horn_black"), scale=1.0)
"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml

from .hair_fit import _centroid, measure

ASSETS = Path(__file__).resolve().parents[1] / "data" / "glasses"
STYLES = {"thin_round": "얇은 동그란 테", "horn_black": "검정 뿔테"}
SCALE_LIMIT = (0.85, 1.15)
SKIN = (1, 10)            # 피부, 코: 테 그림자가 지는 곳
HAIR, GLASSES = 17, 6
LENS_DROP = 0.04          # 렌즈 중심을 눈 중심보다 눈 사이 거리의 이만큼 아래로 (눈동자가 렌즈 가운데보다 조금 위)
LENS_TINT = 0.03          # 렌즈 전체를 이만큼 회색으로 (유리 느낌)
LENS_GLARE = 0.10         # 렌즈 위쪽 사선 반사의 최대 세기
MAX_WIDTH = 1.08          # 테 전체 폭 ≤ 눈 높이 얼굴 폭(hair_fit.measure 의 temple) × 이 값. 넘으면 줄인다 (아이 얼굴)
GRAIN = 0.6               # 사진의 결(노이즈·인쇄 망점)을 테에도 이만큼 입힌다. 테만 너무 깨끗하면 붙인 티가 난다


@lru_cache(maxsize=None)
def load(key: str) -> dict:
    """에셋 → {rgba(float32, 알파 0~1), lens(0~1), centers(2x2), shadow}. 없으면 FileNotFoundError."""
    if key not in STYLES:
        raise KeyError(key)
    png, lens, meta = (ASSETS / f"{key}.png", ASSETS / f"{key}_lens.png", ASSETS / f"{key}.yaml")
    if not (png.exists() and lens.exists() and meta.exists()):
        raise FileNotFoundError(f"안경 사진이 없습니다: {png} → python scripts/make_glasses_assets.py")
    bgra = cv2.imread(str(png), cv2.IMREAD_UNCHANGED).astype(np.float32)
    m = yaml.safe_load(meta.read_text(encoding="utf-8"))
    return {"rgba": np.dstack([bgra[:, :, 2::-1], bgra[:, :, 3] / 255]),
            "lens": cv2.imread(str(lens), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255,
            "centers": np.array(m["lens_centers"], np.float64), "shadow": m["shadow"],
            "width": float(np.ptp(np.nonzero(bgra[:, :, 3].max(0) > 25)[0]))}


def preview(key: str, width: int = 480) -> np.ndarray:
    """화면 카드용 미리보기 RGB (밝은 회색 바탕)."""
    a = load(key)
    rgba = a["rgba"]
    bg = np.full(rgba.shape[:2] + (3,), 236, np.float32)
    bg[a["lens"] > 0.5] = 226
    img = rgba[:, :, :3] * rgba[:, :, 3:] + bg * (1 - rgba[:, :, 3:])
    s = width / img.shape[1]
    return cv2.resize(np.clip(img, 0, 255).astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def eyes(parse: np.ndarray):
    """부위 라벨 → (화면 왼쪽 눈, 오른쪽 눈) 중심. 못 찾으면 None."""
    a, b = _centroid(parse == 4), _centroid(parse == 5)
    if a is None or b is None:
        return None
    return (a, b) if a[0] < b[0] else (b, a)


def _place(asset: dict, el: np.ndarray, er: np.ndarray, scale: float, size, face_w=None) -> tuple:
    """에셋 → 얼굴 크기 (테 RGBA, 렌즈). 먼저 INTER_AREA 로 줄인 뒤 돌리고 옮겨서 가는 테가 깨지지 않게.
    크기: 두 렌즈 중심 거리 = 두 눈 거리. 테 폭이 얼굴 폭(face_w)의 MAX_WIDTH 배를 넘으면 그만큼 줄인다."""
    c = asset["centers"]
    D = float(np.hypot(*(er - el)))
    s = D / float(np.hypot(*(c[1] - c[0])))
    if face_w:
        s = min(s, MAX_WIDTH * face_w / asset["width"])
    s *= scale
    rgba, lens = asset["rgba"], asset["lens"]
    pm = np.dstack([rgba[:, :, :3] * rgba[:, :, 3:], rgba[:, :, 3]])  # 알파를 곱한 색으로 줄여야 테 둘레가 안 번진다
    pm = cv2.resize(pm, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    lens = cv2.resize(lens, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ang = np.degrees(np.arctan2(er[1] - el[1], er[0] - el[0]))
    src_mid = (c[0] + c[1]) / 2 * s
    dst_mid = (el + er) / 2 + np.array([0.0, LENS_DROP * D])
    M = cv2.getRotationMatrix2D(tuple(src_mid), -ang, 1.0)
    M[:, 2] += dst_mid - src_mid
    w, h = size
    pm = cv2.warpAffine(pm, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    lens = cv2.warpAffine(lens, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return pm, lens, D, s * np.hypot(*(c[1] - c[0])) / 2


def apply(img: np.ndarray, parse: np.ndarray, asset: dict, scale: float = 1.0) -> tuple:
    """FFHQ 정렬 얼굴(RGB uint8) + 부위 라벨 → (안경 쓴 얼굴, 정보). 눈을 못 찾으면 ValueError."""
    h, w = img.shape[:2]
    if parse.shape != (h, w):
        parse = cv2.resize(parse, (w, h), interpolation=cv2.INTER_NEAREST)
    e = eyes(parse)
    if e is None:
        raise ValueError("얼굴에서 두 눈을 찾지 못했습니다 (가려졌거나 옆얼굴)")
    el, er = e
    m = measure(parse)
    pm, lens, D, half_span = _place(asset, el, er, scale, (w, h), m and m["temple"])
    pm = cv2.GaussianBlur(pm, (0, 0), 0.6)  # 카메라 흐림 정도만
    fa = pm[:, :, 3].clip(0, 1)
    src = img.astype(np.float32)
    grain = src - cv2.GaussianBlur(src, (0, 0), 1.5)
    frame_rgb = pm[:, :, :3] / np.maximum(fa[..., None], 1e-6) + GRAIN * grain

    # 얼굴 밖으로 나가는 테 끝: 머리카락이 덮으면 지운다 (렌즈 근처는 눈썹 분할 오류에 안 흔들리게 손대지 않음)
    xx = np.arange(w)[None, :].repeat(h, 0)
    mid = (el + er) / 2
    outer = np.clip((np.abs(xx - mid[0]) - (half_span + 0.25 * D)) / (0.1 * D), 0, 1)
    hair = cv2.GaussianBlur((parse == HAIR).astype(np.float32), (0, 0), max(D * 0.01, 1))
    fa = fa * (1 - hair * outer)

    # 테 그림자: 테 모양을 조금 아래로 밀고 흐려서 피부·코에만
    sh = asset["shadow"]
    shadow = cv2.GaussianBlur(fa, (0, 0), max(sh["blur"] * D * 2, 1))
    shadow = np.roll(shadow, int(round(sh["offset"] * D * 2)), axis=0)
    skin = cv2.GaussianBlur(np.isin(parse, SKIN).astype(np.float32), (0, 0), 2)
    shadow = (shadow * skin * sh["opacity"]).clip(0, 1) * (1 - fa)

    # 렌즈: 아주 옅은 회색 + 위쪽 사선 반사 (눈은 그대로 비친다)
    yy = np.arange(h)[:, None].repeat(w, 1)
    diag = ((xx - mid[0]) * 0.6 + (yy - mid[1])) / max(D, 1)
    glare = np.clip(1 - np.abs(diag + 0.45) / 0.18, 0, 1) * LENS_GLARE
    lens_a = (lens * (1 - fa)).clip(0, 1)

    out = src * (1 - shadow)[..., None]
    out = out * (1 - lens_a * LENS_TINT)[..., None] + 128 * (lens_a * LENS_TINT)[..., None]
    out = out + (255 - out) * (lens_a * glare)[..., None]
    out = out * (1 - fa[..., None]) + frame_rgb * fa[..., None]
    out = np.clip(np.round(out), 0, 255).astype(np.uint8)

    touched = (fa > 1e-3) | (shadow > 1e-3) | (lens_a > 1e-3)
    result = img.copy()
    result[touched] = out[touched]   # 원본 유지: 닿지 않은 곳은 입력 그대로
    changed = np.any(result != img, axis=2)
    info = {"had_glasses": bool((parse == GLASSES).mean() > 0.002),
            "eyes": [[round(float(v), 1) for v in el], [round(float(v), 1) for v in er]],
            "changed_ratio": round(float(changed.mean()), 4),
            "width_ratio": round(float((fa.max(0) > 0.1).sum() / m["temple"]), 2) if m else None}
    return result, info
