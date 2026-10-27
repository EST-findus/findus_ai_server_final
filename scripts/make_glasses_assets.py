"""안경 사진 2장(CC0)을 받아 합성용 에셋으로 만든다 (docs/[04] 3장).

    python scripts/make_glasses_assets.py          # 받기 + 만들기 → data/glasses/

만드는 것 (data/glasses/, git 제외)
    <key>.png        안경 테 RGBA (정면, 배경 투명)
    <key>_lens.png   렌즈 영역 마스크 (0/255)
    <key>.yaml       두 렌즈 중심 좌표, 그림자 세기, 출처
    _src/            받은 원본 사진

thin_round  얇은 동그란 테: Wikimedia Commons "Round glasses pixabay.jpg" (CC0). 비스듬히 찍은 금속 라운드 테라서
            거의 정면인 앞쪽 렌즈 테(+ 경첩)만 잘라 타원을 원으로 펴고, 좌우 대칭으로 두 개 놓은 뒤 브리지·코받침을 그린다.
horn_black  검정 뿔테: rawpixel "Eye glasses png illustration" (CC0, 정면). 체크무늬 배경을 지우고 렌즈를 찾고,
            단색이라 아세테이트 광택(위 테 하이라이트)을 조금 더한다.
"""
import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "glasses"
UA = {"User-Agent": "Mozilla/5.0 (findus glasses assets)"}

SOURCES = {
    "thin_round": {
        "file": "round_glasses_pixabay.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/8/8a/Round_glasses_pixabay.jpg",
        "page": "https://commons.wikimedia.org/wiki/File:Round_glasses_pixabay.jpg",
        "license": "CC0 1.0 (Pixabay 원본 https://pixabay.com/en/glasses-fluke-angel-therapy-415257/)",
    },
    "horn_black": {
        "file": "rawpixel_9069278.png",
        "url": "https://images.rawpixel.com/image_png_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIzLTAxL2pvYjk2Mi0xMDZhLnBuZw.png",
        "page": "https://www.rawpixel.com/image/9069278/png-white-background-people",
        "license": "CC0 1.0",
    },
}


def fetch(key: str) -> np.ndarray:
    src = SOURCES[key]
    path = OUT / "_src" / src["file"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"받기: {src['url']}")
        with urllib.request.urlopen(urllib.request.Request(src["url"], headers=UA), timeout=60) as r:
            path.write_bytes(r.read())
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"이미지를 읽지 못했습니다: {path}")
    return img


def unmix(img: np.ndarray, alpha: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """배경과 섞인 가장자리 색에서 배경을 빼 테 본래 색을 되찾는다."""
    a = np.maximum(alpha, 0.05)[..., None]
    return np.clip((img.astype(np.float32) - (1 - a) * bg) / a, 0, 255)


def over(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    """RGBA(0~255, 알파 곱하지 않은 색) 두 장을 겹친다: src 가 위."""
    sa, da = src[:, :, 3:] / 255, dst[:, :, 3:] / 255
    oa = sa + da * (1 - sa)
    rgb = (src[:, :, :3] * sa + dst[:, :, :3] * da * (1 - sa)) / np.maximum(oa, 1e-6)
    return np.dstack([rgb, oa * 255])


def save(key: str, rgba: np.ndarray, lens: np.ndarray, centers, shadow: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"{key}.png"), rgba)
    cv2.imwrite(str(OUT / f"{key}_lens.png"), lens)
    src = SOURCES[key]
    meta = {"key": key, "lens_centers": [[round(float(x), 1), round(float(y), 1)] for x, y in centers],
            "shadow": shadow, "source": {k: src[k] for k in ("page", "license")}}
    (OUT / f"{key}.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"만듦: {OUT / key}.png {rgba.shape[1]}x{rgba.shape[0]}, 렌즈 중심 {meta['lens_centers']}")


# ---------------------------------------------------------------- 얇은 동그란 테

def make_thin_round():
    img = fetch("thin_round")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gold = (hsv[:, :, 1] > 50).astype(np.uint8)

    # 앞쪽 렌즈 테: 사진 왼쪽 큰 원. 테 바깥쪽 점들로 타원을 맞춘다
    box = (800, 490, 1310, 1080)  # x0, y0, x1, y1
    sub = np.zeros_like(gold)
    sub[box[1]:box[3], box[0]:box[2]] = gold[box[1]:box[3], box[0]:box[2]]
    ys, xs = np.nonzero(sub)
    c0 = np.array([xs.mean(), ys.mean()])
    r = np.hypot(xs - c0[0], ys - c0[1])
    keep = r > 0.85 * np.percentile(r, 90)  # 렌즈 안 코받침·비친 다리 빼기
    pts = np.stack([xs[keep], ys[keep]], 1).astype(np.float32)
    for _ in range(3):  # 다리·경첩 점이 끌어당기지 않게: 맞춘 타원에서 먼 점을 빼고 다시 맞춘다
        (cx, cy), (ax1, ax2), ang = cv2.fitEllipse(pts)
        t = np.deg2rad(ang)
        u = (pts - [cx, cy]) @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        rn = np.hypot(u[:, 0] / (ax1 / 2), u[:, 1] / (ax2 / 2))
        pts = pts[np.abs(rn - 1) < 0.04]

    # 타원 → 반지름 R 원 (경첩까지 들어가게 여유)
    R, size = 300, 820
    t = np.deg2rad(ang)
    rot = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])   # 타원 축으로
    sc = np.diag([2 * R / ax1, 2 * R / ax2])
    A = rot.T @ sc @ rot
    M = np.hstack([A, (np.array([size / 2, size / 2]) - A @ np.array([cx, cy]))[:, None]])

    bg = np.array([234, 234, 234], np.float32)
    alpha = np.clip((bg[0] - img[:, :, 0].astype(np.float32)) / 80.0, 0, 1)  # 파랑 채널: 금색은 낮고 흰 배경은 높다
    alpha *= cv2.dilate(gold, np.ones((5, 5), np.uint8)).astype(np.float32)
    color = unmix(img, alpha, bg)
    # 흰 배경 반사로 바랜 금색을 조금 짙게 (채도 ×1.3, 밝기 ×0.88)
    hsv_c = cv2.cvtColor(np.clip(color, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_c[:, :, 1] = np.clip(hsv_c[:, :, 1] * 1.3, 0, 255)
    hsv_c[:, :, 2] *= 0.88
    color = cv2.cvtColor(hsv_c.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    warp = lambda a, interp=cv2.INTER_CUBIC: cv2.warpAffine(a, M, (size, size), flags=interp,
                                                         borderMode=cv2.BORDER_CONSTANT)
    a = warp(alpha).clip(0, 1)
    col = warp(color)

    # 테 띠 + 경첩(왼쪽 9시 방향)만 남긴다
    yy, xx = np.mgrid[:size, :size]
    rr = np.hypot(xx - size / 2, yy - size / 2)
    band = np.abs(rr - R) < 22
    hinge = (xx < size / 2 - R + 10) & (xx > size / 2 - R - 70) & (np.abs(yy - size / 2 + 55) < 32)
    a *= (band | hinge)
    rim_px = a > 0.6
    rim_th = float(np.median([np.count_nonzero(rim_px[size // 2 - R - 30:size // 2 - R + 30, size // 2]) or 8]))

    # 두 렌즈: 왼쪽(사진 그대로) + 오른쪽(좌우 반전). 렌즈 중심 거리 = 지름 + 브리지
    gap = int(0.42 * 2 * R)
    d = 2 * R + gap
    W, H = d + size, size
    rgba = np.zeros((H, W, 4), np.float32)
    left = np.dstack([col, a * 255])
    rgba[:, :size] = left
    rgba[:, d:d + size] = over(rgba[:, d:d + size], left[:, ::-1])
    c1, c2 = (size / 2, size / 2), (d + size / 2, size / 2)

    # 브리지와 코받침 (금색: 테에서 가장 많은 색)
    gold_bgr = np.median(col[a > 0.8], axis=0)
    hi = np.clip(gold_bgr * 1.25 + 20, 0, 255)
    ss = 4  # 크게 그려 줄여서 매끄럽게
    lay = np.zeros((H * ss, W * ss, 4), np.uint8)
    y_b = c1[1] - 0.30 * R
    x0, x1 = c1[0] + R * np.cos(np.arcsin(0.30)) - 4, c2[0] - R * np.cos(np.arcsin(0.30)) + 4
    pts = np.array([[x, y_b - 0.18 * R * np.sin(np.pi * (x - x0) / (x1 - x0))] for x in np.linspace(x0, x1, 40)])
    th = max(int(rim_th * 0.9), 6)
    cv2.polylines(lay, [np.int32(pts * ss)], False, (*[int(v) for v in gold_bgr], 255), th * ss, cv2.LINE_AA)
    cv2.polylines(lay, [np.int32((pts - [0, th * 0.25]) * ss)], False, (*[int(v) for v in hi], 255),
                  max(th // 3, 2) * ss, cv2.LINE_AA)
    for side, cx_ in ((1, c1[0] + 0.62 * R), (-1, c2[0] - 0.62 * R)):  # 코받침: 투명 실리콘 + 가는 금속 팔
        arm = np.array([[cx_ + side * 0.16 * R, c1[1] - 0.05 * R], [cx_ + side * 0.08 * R, c1[1] + 0.12 * R]])
        cv2.polylines(lay, [np.int32(arm * ss)], False, (*[int(v) for v in gold_bgr], 255), 3 * ss, cv2.LINE_AA)
        cv2.ellipse(lay, (int(cx_ * ss), int((c1[1] + 0.22 * R) * ss)), (int(0.07 * R * ss), int(0.13 * R * ss)),
                    side * -15, 0, 360, (225, 228, 232, 70), -1, cv2.LINE_AA)
    lay = cv2.resize(lay, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    lay[:, :, :3] /= np.maximum(lay[:, :, 3:] / 255, 1e-6)  # 줄이며 섞인 투명 검정 빼기
    rgba = over(rgba, np.clip(lay, 0, 255))

    lens = np.zeros((H, W), np.uint8)
    for c in (c1, c2):
        cv2.circle(lens, (int(c[0]), int(c[1])), int(R - rim_th / 2), 255, -1, cv2.LINE_AA)
    save("thin_round", np.clip(rgba, 0, 255).astype(np.uint8), lens, (c1, c2),
         {"opacity": 0.22, "blur": 0.025, "offset": 0.03})


# ---------------------------------------------------------------- 검정 뿔테

def make_horn_black():
    img = fetch("horn_black")
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    frame = (g < 110).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(frame)
    frame = (lab == 1 + int(np.argmax(st[1:, 4]))).astype(np.uint8)      # 가장 큰 덩어리 = 테

    # 테 안의 구멍: 작은 것(리벳)은 테로 채우고, 큰 두 개는 렌즈
    holes = 1 - frame
    n, lab, st, cen = cv2.connectedComponentsWithStats(holes)
    border = {lab[0, 0], lab[0, -1], lab[-1, 0], lab[-1, -1]}
    inner = [i for i in range(1, n) if i not in border]
    lenses = sorted(sorted(inner, key=lambda i: -st[i, 4])[:2], key=lambda i: cen[i][0])
    rivets = [i for i in inner if i not in lenses]
    solid = frame.copy()
    for i in rivets:
        solid[lab == i] = 1

    # 가장자리 알파: 체크무늬(밝기 230~255)와 테(약 34) 사이 비율
    edge = cv2.dilate(solid, np.ones((3, 3), np.uint8)) - cv2.erode(solid, np.ones((3, 3), np.uint8))
    alpha = solid.astype(np.float32)
    soft = np.clip((235 - g) / (235 - 34), 0, 1)
    alpha[edge > 0] = soft[edge > 0]
    for i in rivets:
        alpha[lab == i] = 1
    bg = np.array([235, 235, 235], np.float32)
    col = unmix(img, alpha, bg)
    col[solid > 0] = img[solid > 0]  # 테 안쪽(리벳 포함)은 원래 색

    # 아세테이트 광택: 위쪽 테두리를 따라 밝은 띠, 아래로 갈수록 아주 조금 어둡게
    H, W = g.shape
    top_edge = np.clip(solid.astype(np.float32) - np.roll(solid, 7, axis=0), 0, 1)
    gloss = cv2.GaussianBlur(top_edge, (0, 0), 3) * 55
    yy = np.mgrid[:H, :W][0].astype(np.float32)
    ys = np.nonzero(solid)[0]
    shade = 1 - 0.12 * np.clip((yy - ys.min()) / (ys.max() - ys.min()), 0, 1)
    body = (solid > 0) & ~np.isin(lab, rivets)
    col[body] = np.clip(col[body] * shade[body, None] + gloss[body, None], 0, 255)

    lens = np.isin(lab, lenses).astype(np.uint8) * 255
    centers = [cen[i] for i in lenses]
    rgba = np.dstack([col, alpha * 255]).astype(np.uint8)
    # 2배로 키워 둔다 (얼굴에 맞출 때 줄이기만 하게)
    rgba = cv2.resize(rgba, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    lens = cv2.resize(lens, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    save("horn_black", rgba, lens, [(x * 2, y * 2) for x, y in centers],
         {"opacity": 0.38, "blur": 0.03, "offset": 0.04})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(SOURCES), help="하나만 만들기")
    args = ap.parse_args()
    for key, fn in (("thin_round", make_thin_round), ("horn_black", make_horn_black)):
        if not args.only or args.only == key:
            fn()


if __name__ == "__main__":
    main()
