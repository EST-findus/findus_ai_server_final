"""머리 크기 맞추기: 참조 사진의 머리를 대상 얼굴 크기에 맞춘다 (HairFastGAN 에 넣기 전).

HairFastGAN 입력은 모두 FFHQ 정렬(눈·입 기준)이라 얼굴 '폭'과 이마 높이는 사람마다 다르다. 얼굴이 좁은 참조의 머리를
얼굴이 넓은 사람에게 그대로 입히면 머리가 작아 보이고 관자놀이가 비며, 반대면 머리가 붕 뜬다. 그래서
  1) 재기: 얼굴 분할(BiSeNet 19부위) 라벨에서 눈·입 위치와 얼굴 폭(관자놀이·광대·턱)을 픽셀로 잰다.
  2) 고르기: 같은 스타일 참조 여러 장 중 얼굴 폭 비율이 대상과 가장 비슷한 것을 고른다 (pick).
  3) 맞추기: 고른 참조를 눈 중심 기준으로 늘이거나 줄여(배율 = 대상 얼굴 폭 / 참조 얼굴 폭) 눈 위치를 대상에 겹친다 (fit).
     너무 늘리면 머리결이 뭉개져서 0.85~1.2배로 제한한다. 사용자가 고른 '머리 크기' 배율을 그 위에 곱한다.

    m = measure(parser.parse(face_rgb))        # parser = src.hair_color.HairMask
    i = pick(target_m, [ref1_m, ref2_m, ...])
    ref_fit, s = fit(ref_rgb, ref_m, target_m, user_scale=1.0)
"""
import cv2
import numpy as np

EYES = (4, 5)
MOUTH = (11, 12, 13)
FACE = (1, 2, 3, 4, 5, 6, 10, 11, 12, 13)   # 피부·눈썹·눈·안경·코·입 (귀 제외)
HAIR, EARS = 17, (7, 8)
LIMIT = (0.85, 1.2)        # 자동 배율 범위
DARK_BG_LIMIT = 0.15       # 자동 고르기에서 이보다 배경이 어두운 참조는 건너뛴다 (dark_bg)
USER_LIMIT = (0.8, 1.25)   # 사용자 '머리 크기' 배율 범위


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()]) if len(xs) > 20 else None


def _width(mask, y, band):
    """y 주변 행들에서 마스크 가로 폭(px)의 중앙값."""
    h = mask.shape[0]
    ws = []
    for r in range(int(max(y - band, 0)), int(min(y + band + 1, h))):
        xs = np.nonzero(mask[r])[0]
        if len(xs) > 2:
            ws.append(xs.max() - xs.min())
    return float(np.median(ws)) if ws else None


def measure(lab: np.ndarray):
    """부위 라벨 지도 → 크기 지표 (픽셀). 눈이나 입을 못 찾으면 None.
    eye: 두 눈 중심, D: 눈 사이 거리, temple/cheek/jaw: 눈 높이·눈과 입 사이·입 아래의 얼굴 폭,
    head_top: 눈에서 머리(또는 피부) 꼭대기까지 높이 (정렬 사진은 머리가 위 끝에 닿는 일이 많아 확인용으로만),
    hair_w: 이마 높이에서 머리+얼굴+귀 폭. 안경이 눈을 가려 눈을 못 찾으면 None (그때는 맞추지 않는다)."""
    el, er = _centroid(lab == EYES[0]), _centroid(lab == EYES[1])
    mouth = _centroid(np.isin(lab, MOUTH))
    if el is None or er is None or mouth is None:
        return None
    eye = (el + er) / 2
    D = float(np.hypot(*(el - er)))
    em = float(mouth[1] - eye[1])
    face = np.isin(lab, FACE)
    band = max(2, int(0.04 * D))
    # 턱 폭은 입 높이에서 잰다 (입 아래로 내려가면 턱 끝에 걸려 너무 좁게 나온다)
    temple, cheek, jaw = (_width(face, y, band) for y in (eye[1], eye[1] + 0.5 * em, mouth[1]))
    if not (temple and cheek):
        return None
    head = np.isin(lab, FACE + EARS + (HAIR, 18))
    cx = int(eye[0])
    col = head[:, max(cx - int(0.3 * D), 0):cx + int(0.3 * D)].any(1)
    top = int(np.argmax(col)) if col.any() else int(eye[1])
    return {"eye": eye.tolist(), "D": D, "em": em, "temple": temple, "cheek": cheek, "jaw": jaw or cheek,
            "head_top": float(eye[1] - top), "hair_w": _width(head, eye[1] - 0.6 * em, band) or temple}


def face_ratio(tgt: dict, ref: dict) -> float:
    """대상 얼굴 폭 / 참조 얼굴 폭 (관자놀이·광대 평균, 기하평균)."""
    return float(np.sqrt(tgt["temple"] / ref["temple"] * tgt["cheek"] / ref["cheek"]))


def dark_bg(lab: np.ndarray, img_rgb: np.ndarray) -> float:
    """머리 둘레 배경(머리 바깥 약 15px, 어깨 위쪽만) 중 머리만큼 어두운 픽셀 비율 (0~1).
    배경이 어두우면(검은 의자·문·물체) HairFast 가 배경까지 머리로 읽어 머리가 옆으로 부풀거나 덩어리가 붙는다
    (2026-10-19, docs/[20] 1장: 덮머 참조 8장 중 0.15 넘는 3장이 모두 일그러짐)."""
    lab = cv2.resize(lab, (512, 512), interpolation=cv2.INTER_NEAREST)
    img = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA)
    hair = (lab == HAIR).astype(np.uint8)
    if hair.sum() < 100:
        return 0.0
    ring = (cv2.dilate(hair, np.ones((31, 31), np.uint8)) > 0) & (lab == 0)
    ring[300:] = False
    if ring.sum() == 0:
        return 0.0
    L = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[..., 0].astype(np.float32) * 100 / 255
    return float(((L < L[hair > 0].mean() + 15) & ring).sum() / ring.sum())


def quality(lab: np.ndarray, img_rgb: np.ndarray) -> dict:
    """참조 사진 품질 지표 (얼굴 분할 라벨 기준). 참조 고르기·선정에 쓴다 (scripts/hair_ref_quality.py, docs/[20] 2장).
    dark_bg: 머리 둘레 배경이 머리만큼 어두운 비율 / sharp: 머리 안 결의 세기 (낮으면 결 없는 검은 덩어리)
    top_cut: 머리가 사진 위 끝에 닿는 비율 / roll·yaw: 고개 기울기(도)·돌림(눈 사이 거리 대비 코 치우침)
    sym: 눈 위 머리의 좌우 대칭 (IoU) / frag: 가장 큰 머리 조각 밖 머리 비율 / hat: 모자 비율."""
    lab = cv2.resize(lab, (512, 512), interpolation=cv2.INTER_NEAREST)
    img = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA)
    hair = (lab == HAIR).astype(np.uint8)
    q = {"dark_bg": dark_bg(lab, img), "top_cut": float(hair[0].mean()), "hat": float((lab == 18).mean())}
    n, _, st, _ = cv2.connectedComponentsWithStats(hair)
    q["frag"] = float(1 - st[1:, 4].max() / hair.sum()) if n > 1 else 1.0
    m = measure(lab)
    if m is None:
        q.update(roll=90.0, yaw=1.0, sym=0.0)
    else:
        el, er, nose = _centroid(lab == EYES[0]), _centroid(lab == EYES[1]), _centroid(lab == 10)
        a = abs(np.degrees(np.arctan2(er[1] - el[1], er[0] - el[0])))
        q["roll"] = float(min(a, 180 - a))   # 왼눈·오른눈 라벨 순서와 상관없이
        q["yaw"] = float(abs(nose[0] - m["eye"][0]) / m["D"]) if nose is not None else 1.0
        cx, ey = m["eye"][0], int(m["eye"][1])
        up = hair[:ey].astype(bool)
        mir = np.roll(up[:, ::-1], int(round(2 * cx - 511)), axis=1)
        q["sym"] = float((up & mir).sum() / max((up | mir).sum(), 1))
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.0)  # 망점 잡음은 눌러서
    inner = cv2.erode(hair, np.ones((5, 5), np.uint8)) > 0
    q["sharp"] = float(cv2.Laplacian(g, cv2.CV_32F)[inner].std()) if inner.sum() > 500 else 0.0
    return q


# 탈락 규칙 (2026-10-19, 덮머 8장을 같은 얼굴에 입혀 본 결과로 정함. docs/[20] 1-4·2장)
QUALITY_RULES = {
    "dark_bg": lambda q: q["dark_bg"] > DARK_BG_LIMIT,             # 배경까지 머리로 읽어 부풂·덩어리
    "sharp": lambda q: q["sharp"] < 2.0,                           # 결 없는 덩어리 → 헬멧 모양
    "pose": lambda q: q["yaw"] > 0.06 or q["roll"] > 4,            # 돌리거나 기울인 얼굴
    "top_cut": lambda q: q["top_cut"] > 0.45,                      # 머리 윗부분이 많이 잘림
    "frag": lambda q: q["frag"] > 0.1,                             # 머리 분할이 조각남 (가림)
    "hat": lambda q: q["hat"] > 0.005,
}


def quality_fails(q: dict) -> list:
    return [k for k, f in QUALITY_RULES.items() if f(q)]


def quality_score(q: dict) -> float:
    """통과한 참조끼리 순위 (클수록 좋음): 좌우 대칭 + 결 선명 − 위 잘림 − 배경 어두움."""
    return q["sym"] + 0.08 * min(q["sharp"], 6) - 0.5 * q["top_cut"] - q["dark_bg"]


def pick(tgt: dict, refs: list, bad=None) -> int:
    """참조 지표 목록 중 대상과 얼굴 폭 비율·턱 비율이 가장 가까운 것의 번호. 지표가 None 인 참조는 건너뛴다.
    bad[i] 가 참이면(배경이 어두운 참조 등) 건너뛴다. 모두 bad 면 bad 를 무시한다."""
    def cost(r):
        if r is None:
            return np.inf
        return abs(np.log(face_ratio(tgt, r))) + 0.5 * abs(np.log((tgt["jaw"] / tgt["cheek"]) / (r["jaw"] / r["cheek"])))
    costs = np.array([cost(r) for r in refs])
    if bad is not None and not all(bad[i] or not np.isfinite(c) for i, c in enumerate(costs)):
        costs = np.where(bad, np.inf, costs)
    return int(np.argmin(costs))


def fit(ref_rgb: np.ndarray, ref: dict, tgt: dict, user_scale: float = 1.0, auto: bool = True):
    """참조 사진을 눈 중심 기준으로 늘이고 줄여 대상 얼굴 크기·눈 위치에 맞춘다. 반환 (맞춘 사진, 쓴 배율)."""
    s = float(np.clip(face_ratio(tgt, ref), *LIMIT)) if auto else 1.0
    s *= float(np.clip(user_scale, *USER_LIMIT))
    if abs(s - 1) < 0.01 and np.hypot(*(np.subtract(ref["eye"], tgt["eye"]))) < 2:
        return ref_rgb, s
    (rx, ry), (tx, ty) = ref["eye"], tgt["eye"]
    M = np.float32([[s, 0, tx - s * rx], [0, s, ty - s * ry]])
    h, w = ref_rgb.shape[:2]
    out = cv2.warpAffine(ref_rgb, M, (w, h), flags=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA,
                         borderMode=cv2.BORDER_REPLICATE)
    return out, s


def overlay(img: np.ndarray, m: dict) -> np.ndarray:
    """확인용 그림: 잰 위치(눈 높이 폭, 광대 폭, 턱 폭, 머리 꼭대기)를 선으로 그린다."""
    out = img.copy()
    if m is None:
        return out
    ex, ey = m["eye"]
    em = m["em"]
    for y, w, c in ((ey, m["temple"], (255, 0, 0)), (ey + 0.5 * em, m["cheek"], (0, 255, 0)),
                    (ey + em, m["jaw"], (0, 128, 255)), (ey - 0.6 * em, m["hair_w"], (255, 0, 255))):
        cv2.line(out, (int(ex - w / 2), int(y)), (int(ex + w / 2), int(y)), c, max(2, img.shape[0] // 256))
    cv2.line(out, (int(ex), int(ey)), (int(ex), int(ey - m["head_top"])), (255, 255, 0), max(2, img.shape[0] // 256))
    return out
