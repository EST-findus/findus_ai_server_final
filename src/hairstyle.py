"""후처리: HairFastGAN 으로 머리 모양·색 바꾸기. 입력은 FFHQ 정렬 1024 얼굴(RGB), 출력도 같은 형식.
나이 변환 파이프라인(src/pipeline.py)과 따로 돈다. 이미 만든 결과에 얹기만 하므로 참조를 바꿔도 나이 변환을 다시 하지 않는다.

    post = HairPost()                                            # 여러 장 + 참조 정렬 + 신원 점수
    outs = post.run_many({"30": img30, "40": img40}, (shape_rgb, True), evaluators=["arcface"])
    faces = post.align({"me": any_photo_rgb})                    # 정렬 안 된 사진은 먼저 정렬 (/hair 화면)
    post.free()

얼굴 지키기 (keep_face, 기본 켬): HairFastGAN 은 머리만이 아니라 사진 전체를 다시 그려서 얼굴이 조금 바뀐다
(바꾸기 전과 ArcFace 0.62~0.85). 전문가가 고친 몽타주의 얼굴이 흐려지지 않게, 얼굴 분할(BiSeNet)로
'입력의 얼굴 부위(피부·눈썹·눈·안경·귀·코·입) 이면서 결과에서 머리가 아닌 곳'은 입력 사진을 그대로 쓰고
나머지(머리·배경·목·옷)만 HairFastGAN 결과를 쓴다. 경계는 부드럽게 섞는다. 새 앞머리가 이마를 덮으면
그 부분은 결과(머리)를 쓰고, 원래 앞머리가 없어져 드러난 이마는 결과가 그린 피부를 쓴다.

머리 크기 맞추기 (src/hair_fit.py): 같은 스타일 참조가 여러 장이면(candidates) 대상마다 얼굴 폭이 가장 비슷한
참조를 고르고 (품질 규칙에 걸리는 참조는 건너뜀, hair_fit.quality), fit=True 면 고른 참조를 대상 얼굴 폭에 맞춰 늘이고 줄인 뒤(0.85~1.2배) HairFastGAN 에 넣는다.
scale 은 사용자가 고른 '머리 크기' 배율 (그 위에 곱한다).

    styler = HairStyler()                                        # 모델만 (정렬된 사진 한 장씩)
    out = styler(face_rgb, shape_rgb, color_rgb)   # color 를 생략하면 shape 사진의 색
    styler.free()

SAM 과 HairFast 가 같은 패키지 이름(models, utils, configs …)을 쓰므로, HairFast 쪽 모듈은 따로 보관했다가
호출할 때만 sys.modules 에 넣는다. 그래서 한 프로세스(웹 서버)에서 둘을 번갈아 써도 된다.
GPU 6GB 기준 로드 12~18초, 1장 13~26초 (VRAM 이 모자라 공유 메모리로 넘침, docs/[04]_후처리_계획.md).
"""
import importlib.util
import os
import sys
import types
from contextlib import contextmanager

import cv2
import numpy as np

from .align import arcface_align, ffhq_align
from . import hair_fit
from .hair_color import HAIR, HairMask
from .paths import THIRD_PARTY
from .pipeline import Pipeline
from .stages.evaluate import cosine

HF_DIR = THIRD_PARTY / "HairFastGAN"
_SHARED = ("models", "utils", "configs", "criteria", "datasets")  # SAM 에도 있는 최상위 패키지 이름
# HairFast 쪽 모듈 보관함. 프로세스가 끝날 때까지 유지한다 (코드뿐이라 GPU 메모리와 무관).
# 모델을 내렸다 다시 올릴 때 모듈을 새로 import 하면, 이름 밖에 남은 옛 모듈의 클래스와 섞여 SEAN 의
# issubclass 검사가 실패한다.
_MODULES = {}
FACE_PARTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)  # 피부, 눈썹, 눈, 안경, 귀, 귀걸이, 코, 입, 입술 (hair_color.parse)


def composite_face(before: np.ndarray, after: np.ndarray, parser, feather: int = 10) -> np.ndarray:
    """before(입력)의 얼굴 부위 중 after 에서 머리가 아닌 곳은 before 를, 나머지는 after 를 쓴다.
    feather: 512 기준 경계 흐림 폭(px). 크기는 after 에 맞춘다. parser = HairMask (parse 메서드)."""
    if before.shape != after.shape:
        before = cv2.resize(before, after.shape[1::-1], interpolation=cv2.INTER_AREA)
    lb = parser.parse(before)
    w = after.shape[1]
    # 앞머리 끝의 가는 머리카락·그림자는 '피부'로 분류되기 쉽다 → 입력 머리 경계 아래 피부 띠는 입력에서 가져오지 않는다
    keep = np.isin(lb, FACE_PARTS) & ~_fringe_band(lb) & (parser.parse(after) != HAIR)
    keep = cv2.erode(keep.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(1, w // 512))  # 머리 가장자리 픽셀이 새지 않게
    k = max(1, int(feather * w / 512)) * 2 + 1
    a = cv2.GaussianBlur(keep.astype(np.float32), (k, k), 0)[..., None]
    return (before * a + after * (1 - a)).round().clip(0, 255).astype(np.uint8)


def _fringe_band(lab: np.ndarray, down: int = 30) -> np.ndarray:
    """머리 경계 바로 아래 '피부' 띠 (1024 기준 down px). 앞머리 끝의 가는 머리카락·그림자가 피부로 분류되는 곳.
    눈썹·눈은 넣지 않는다 (원래 눈썹을 지키려고)."""
    h, w = lab.shape
    hair = (lab == HAIR).astype(np.uint8)
    d = max(1, int(down * w / 1024))
    kernel = np.zeros((2 * d + 1, 2 * d // 3 + 1), np.uint8)
    kernel[:d + 1] = 1                       # 아래쪽으로만 넓힌다 (dilate 는 커널을 뒤집어 적용)
    return (cv2.dilate(hair, kernel) > 0) & (lab == 1)


def _fill_from(img: np.ndarray, src: np.ndarray, sigmas) -> np.ndarray:
    """src 픽셀의 색만으로 빈 곳을 채운다. 여러 크기 블러를 더해 나눠서, 가까운 원천일수록 무게가 크고 경계 줄무늬가 없다."""
    f, s = img.astype(np.float32), src.astype(np.float32)
    num = np.zeros(img.shape, np.float32)
    den = np.zeros(src.shape, np.float32)
    for sig in sigmas:
        num += cv2.GaussianBlur(f * s[..., None], (0, 0), sig)
        den += cv2.GaussianBlur(s, (0, 0), sig)
    return num / np.maximum(den, 1e-6)[..., None]


def _skin_texture(img: np.ndarray, skin: np.ndarray, cx: float, eye_y: float, fw: float) -> np.ndarray:
    """본인 볼의 피부결(고주파)을 뒤집어 가며 타일로 깐 것. 칠한 이마가 매끈한 판처럼 보이지 않게 한다.
    볼 조각이 피부가 아니면(가림·그림자) 반대쪽 볼, 둘 다 안 되면 결 없음."""
    h, w = skin.shape
    hp = img.astype(np.float32) - cv2.GaussianBlur(img.astype(np.float32), (0, 0), 2.0 * w / 1024)
    n = max(16, int(0.16 * fw))
    for side in (-1, 1):
        x0, y0 = int(cx + side * 0.24 * fw - n / 2), int(eye_y + 0.22 * fw)
        if 0 <= x0 and x0 + n <= w and y0 + n <= h and skin[y0:y0 + n, x0:x0 + n].mean() > 0.95:
            tile = hp[y0:y0 + n, x0:x0 + n]
            block = np.vstack([np.hstack([tile, tile[:, ::-1]]), np.hstack([tile[::-1], tile[::-1, ::-1]])])  # 이음매 없게
            reps = (h // (2 * n) + 1, w // (2 * n) + 1, 1)
            return np.tile(block, reps)[:h, :w]
    return np.zeros_like(hp)


def remove_hair(img: np.ndarray, parser) -> np.ndarray:
    """대상의 원래 머리를 지운 사진 (HairFastGAN 입력용). 이마 쪽은 주변 피부색, 나머지는 주변 배경색으로 칠한다.
    원래 머리가 남아 있으면 HairFast 가 새 머리 밖(특히 앞머리를 걷어 낸 이마)에 옛 머리카락을 잔머리처럼 남긴다.
    칠은 거칠어도 된다: HairFast 가 StyleGAN 으로 사진 전체를 다시 그리고, 얼굴 부위는 composite_face 가 입력으로 되돌린다."""
    lab = parser.parse(img)
    h, w = lab.shape
    k = w / 1024
    hair = cv2.dilate((lab == HAIR).astype(np.uint8), np.ones((int(25 * k) | 1,) * 2, np.uint8)) > 0  # 경계 그림자까지
    hair = (hair | _fringe_band(lab)) & ~np.isin(lab, (2, 3, 4, 5, 6))   # 앞머리 끝까지, 눈썹·눈은 남김
    face = np.isin(lab, (1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13))
    eyes = np.nonzero(np.isin(lab, (4, 5)))[0]
    if hair.sum() < 0.01 * h * w or len(eyes) < 20:
        return img
    eye_y = eyes.mean()
    cols = np.nonzero(face[int(eye_y)])[0]
    if len(cols) < 2:
        return img
    cx, fw = (cols.min() + cols.max()) / 2, cols.max() - cols.min()
    shrink = np.ones((int(7 * k) | 1,) * 2, np.uint8)
    skin = (cv2.erode((lab == 1).astype(np.uint8), shrink) > 0) & ~hair
    bg = (cv2.erode((lab == 0).astype(np.uint8), shrink) > 0) & ~hair
    sig = [s * k for s in (15, 40, 100, 250)]
    yy, xx = np.mgrid[:h, :w]
    # 이마·정수리 자리: 눈 높이 위, 얼굴 폭 기준 타원 (민머리 두상 모양) → 피부, 그 밖 → 배경
    dome = (((xx - cx) / (0.46 * fw)) ** 2 + ((yy - eye_y) / (0.62 * fw)) ** 2 < 1) & (yy < eye_y)
    wd = cv2.GaussianBlur(dome.astype(np.float32), (0, 0), 12 * k)[..., None]
    skin_fill = _fill_from(img, skin, sig) + _skin_texture(img, skin, cx, eye_y, fw)
    fill = wd * skin_fill + (1 - wd) * (_fill_from(img, bg, sig) if bg.any() else img)
    a = cv2.GaussianBlur(hair.astype(np.float32), (0, 0), 3 * k)[..., None]
    return (a * fill + (1 - a) * img).round().clip(0, 255).astype(np.uint8)


def tone_forehead(after: np.ndarray, lb: np.ndarray, la: np.ndarray) -> np.ndarray:
    """원래 머리(앞머리)였다가 피부가 된 이마의 밝기·색을 바로 아래 진짜 피부에 맞추고, 새 머리선 쪽으로 그늘을 넣는다.
    remove_hair 로 칠한 이마는 HairFast 가 거의 그대로 옮겨서, 그냥 두면 밝고 매끈한 판처럼 보인다.
    lb, la = 입력·결과의 부위 라벨."""
    h, w = la.shape
    R = ((lb == HAIR) | _fringe_band(lb)) & (la == 1)
    if R.sum() < 0.002 * h * w:
        return after
    yy = np.mgrid[:h, :w][0]
    yb = np.nonzero(R)[0].max()
    ref = (la == 1) & (lb == 1) & ~R & (yy > yb - 0.02 * h) & (yy < yb + 0.06 * h)
    if ref.sum() < 200:
        return after
    lab = cv2.cvtColor(after.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    mr, mR = lab[ref].mean(0), lab[R].mean(0)
    dist = cv2.distanceTransform(1 - (la == HAIR).astype(np.uint8), cv2.DIST_L2, 5)
    shade = 8 * np.exp(-dist / (0.03 * w))          # 머리선 가까울수록 최대 L 8 어둡게
    out = lab - mR + mr
    out[..., 0] -= 2 + shade
    a = cv2.GaussianBlur(R.astype(np.float32), (0, 0), 0.012 * w)[..., None]
    res = cv2.cvtColor(a * out + (1 - a) * lab, cv2.COLOR_Lab2RGB)
    return (res.clip(0, 1) * 255).round().astype(np.uint8)


def mirror_brow(img: np.ndarray, lab_in: np.ndarray, lab_out: np.ndarray = None) -> np.ndarray:
    """앞머리에 가려 한쪽 눈썹이 조각만 남았으면(반대쪽의 절반 미만), 조각을 지우고 반대쪽 눈썹을 눈 중심 기준으로
    좌우 뒤집어 옮겨 그린다. 피부색은 그대로 두고 눈썹의 '어두운 정도'만 옮긴다.
    lab_out(결과의 부위 라벨)을 주면 지울 조각을 결과 기준으로도 잡는다 (입력 라벨만으로는 점이 남았다)."""
    h, w = lab_in.shape
    brows = [lab_in == 2, lab_in == 3]
    eyes = [hair_fit._centroid(lab_in == 4), hair_fit._centroid(lab_in == 5)]
    areas = [b.sum() for b in brows]
    if any(e is None for e in eyes) or min(areas) >= 0.5 * max(areas) or max(areas) < 300:
        return img
    big, small = (0, 1) if areas[0] > areas[1] else (1, 0)
    ex_b, ey_b = eyes[big]
    ex_s, ey_s = eyes[small]
    lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    # 1) 남은 조각 지우기 (작은 영역이라 주변 피부로 메우기)
    stub = brows[small] | (lab_out == (2, 3)[small] if lab_out is not None else False)
    stub = cv2.dilate(stub.astype(np.uint8), np.ones((int(15 * w / 1024) | 1,) * 2, np.uint8))
    out = cv2.inpaint(img, stub, max(3, int(7 * w / 1024)), cv2.INPAINT_TELEA)
    lab_out = cv2.cvtColor(out.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    # 2) 큰 눈썹의 어두운 정도(주변 피부와의 차이)를 뒤집어 옮긴다: x' = ex_s + (ex_b - x), y' = y + (ey_s - ey_b)
    m = cv2.dilate(brows[big].astype(np.uint8), np.ones((int(7 * w / 1024) | 1,) * 2, np.uint8)).astype(np.float32)
    skin_mean = cv2.GaussianBlur(lab, (0, 0), 0.02 * w)
    delta = (lab - skin_mean) * m[..., None]
    M = np.float32([[-1, 0, ex_s + ex_b], [0, 1, ey_s - ey_b]])
    d2 = cv2.warpAffine(delta, M, (w, h))
    m2 = cv2.GaussianBlur(cv2.warpAffine(m, M, (w, h)), (0, 0), 1.5 * w / 1024)[..., None]
    res = lab_out + m2 * d2
    res = cv2.cvtColor(res, cv2.COLOR_Lab2RGB)
    return (res.clip(0, 1) * 255).round().astype(np.uint8)


REF_BG = (0, 15, 16)   # 배경, 목걸이, 옷 (hair_color.parse)
GRAY = 200


def clean_ref(ref: np.ndarray, parser) -> np.ndarray:
    """참조 사진의 배경·옷을 밝은 회색으로 지운다 (머리·얼굴·목은 남김). HairFastGAN 에 넣기 전.
    어두운 배경(의자·문·물체)을 HairFast 가 머리로 읽어 머리가 부풀거나 덩어리·줄무늬가 생기던 것을 막는다.
    얼굴은 지우지 않는다: HairFast 가 얼굴을 기준으로 고개 방향과 머리 모양을 맞추기 때문 (docs/[21] 3장).
    머리 경계는 머리 확률 마스크로 부드럽게 남긴다."""
    keep = (~np.isin(parser.parse(ref), REF_BG)).astype(np.float32)
    keep = cv2.GaussianBlur(np.maximum(keep, parser(ref)), (0, 0), 2 * ref.shape[1] / 1024)[..., None]
    return (keep * ref + (1 - keep) * GRAY).round().clip(0, 255).astype(np.uint8)


def hair_only(ref: np.ndarray, parser) -> np.ndarray:
    """화면 썸네일용: 머리만 원래 사진, 얼굴·목은 회색 실루엣(머리 위치를 알 수 있게), 나머지는 밝은 회색.
    참조 사진 속 사람 얼굴이 화면에 보이지 않게 한다 (AI Hub 사진, 재배포 금지)."""
    lab = parser.parse(ref)
    out = np.full_like(ref, 235)
    out[np.isin(lab, FACE_PARTS + (14,))] = 185
    a = np.clip((parser(ref) - 0.3) / 0.5, 0, 1)   # 머리 확률이 낮은 배경 번짐은 잘라낸다
    a = cv2.GaussianBlur(a, (0, 0), 1.0)[..., None]
    return (a * ref + (1 - a) * out).round().clip(0, 255).astype(np.uint8)


def _pop_shared() -> dict:
    return {k: sys.modules.pop(k) for k in list(sys.modules) if k.split(".")[0] in _SHARED}


class HairStyler:
    def __init__(self, device="cuda"):
        if not (HF_DIR / "pretrained_models").exists():
            raise FileNotFoundError(f"{HF_DIR}/pretrained_models 가 없습니다 → python scripts/try_hairfast.py --download")
        with self._isolated():
            from hair_swap import HairFast, get_parser
            self.net = HairFast(get_parser().parse_args(["--device", device]))

    @contextmanager
    def _isolated(self):
        """HairFast 모듈·경로·작업 폴더(가중치를 상대경로로 읽음)를 잠깐 바꿔 끼운다."""
        saved, path, cwd = _pop_shared(), sys.path[:], os.getcwd()
        sys.modules.update(_MODULES)
        sys.path.insert(0, str(HF_DIR))
        # dlib 은 임의 사진 정렬(align=True)에서만 쓰므로 설치돼 있지 않으면 빈 모듈로 대신한다
        stub = "dlib" not in sys.modules and importlib.util.find_spec("dlib") is None
        if stub:
            sys.modules["dlib"] = types.ModuleType("dlib")
        os.chdir(HF_DIR)
        try:
            yield
        finally:
            os.chdir(cwd)
            if stub:
                sys.modules.pop("dlib", None)
            _MODULES.update(_pop_shared())
            sys.modules.update(saved)
            sys.path[:] = path

    def __call__(self, face_rgb: np.ndarray, shape_rgb: np.ndarray, color_rgb: np.ndarray = None) -> np.ndarray:
        import torch
        tensors = [torch.from_numpy(_to1024(x)).permute(2, 0, 1).contiguous()
                   for x in (face_rgb, shape_rgb, shape_rgb if color_rgb is None else color_rgb)]
        with self._isolated(), torch.no_grad():
            out = self.net(*tensors)
        return out.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()

    def free(self):
        import gc
        import torch
        self.net = None
        gc.collect()
        torch.cuda.empty_cache()


class HairPost:
    """나이 변환 결과 여러 장에 같은 머리 참조를 입힌다. 섹션: 참조 정렬(검출기) → 머리 변환 → 신원 점수(검출기 + 평가망).
    HairFast(5.5GB)는 참조만 바꿔 다시 적용할 때를 위해 free() 전까지 올려 두고, 검출기·평가망은 섹션마다 내린다.
    6GB 에 나이 변환 모델과 같이 안 들어가므로, 나이 변환 전에 free() 를 부른다."""

    def __init__(self, device="cuda", detector: dict = None):
        self.pipe = Pipeline({"device": device, "detector": detector or {"name": "retinaface"}})
        self.styler = None
        self.last_fit = {}   # 마지막 run_many 에서 결과마다 쓴 참조 번호(0 = shape, 1~ = candidates)와 배율

    def run_many(self, images: dict, shape, color=None, evaluators=(), keep_face: bool = True,
                 candidates=None, fit: bool = False, scale: float = 1.0, clean_bg: bool = True,
                 clear_hair: bool = True) -> dict:
        """images = {키: 정렬된 얼굴} → {키: (머리 바꾼 이미지, {평가망: 바꾸기 전과의 코사인})}.
        shape/color = (이미지, 정렬 필요 여부). 직접 올린 사진처럼 정렬 안 된 사진은 검출 → FFHQ 정렬한다.
        color 가 None 이면 shape 사진의 색을 쓴다. evaluators 가 비면 점수를 생략한다.
        keep_face: 얼굴 부위는 입력 사진을 그대로 쓴다. candidates/fit/scale: 머리 크기 맞추기 (모듈 설명 참고).
        clean_bg: 모양 참조의 배경·옷을 회색으로 지운 뒤 넣는다 (clean_ref).
        clear_hair: 대상의 원래 머리를 지운 뒤 넣는다 (remove_hair, 이마·윤곽 잔머리 방지).
        결과마다 쓴 참조·배율은 self.last_fit[키] = {"ref": 번호, "scale": 배율, "ratio": 대상/참조 얼굴 폭}."""
        shape, color = self._align_refs(shape, color)
        refs = [shape] + self._align_list(candidates or [])
        sizing = len(refs) > 1 or fit or abs(scale - 1) > 1e-3
        parser = HairMask(self.pipe.device) if (sizing or keep_face or clean_bg or clear_hair) else None  # 작다 (BiSeNet)
        try:
            # 참조 고르기·품질 판단은 원래 사진으로 (배경이 어두운지도 품질 정보라서), 지우기는 넣기 직전에
            plan = self._plan(images, refs, parser, fit, scale) if sizing else {k: (shape, 0, 1.0, None) for k in images}
            self.last_fit = {k: {"ref": i, "scale": round(s, 3), "ratio": r and round(r, 3)} for k, (_, i, s, r) in plan.items()}
            shapes = {k: clean_ref(_to1024(p[0]), parser) if clean_bg else p[0] for k, p in plan.items()}
            faces = {k: remove_hair(_to1024(img), parser) if clear_hair else img for k, img in images.items()}
            if self.styler is None:
                self.pipe._free()  # 앞에서 내린 모델의 GPU 캐시까지 비운 뒤 올린다
                self.styler = HairStyler(self.pipe.device)
            outs = {k: self.styler(faces[k], shapes[k], color) for k in images}
            if keep_face:
                outs = {k: composite_face(images[k], out, parser) for k, out in outs.items()}
            if clear_hair:  # 칠한 이마를 주변 피부에 맞추고, 앞머리에 가렸던 눈썹을 채운다
                for k in outs:
                    lb = parser.parse(_to1024(images[k]))
                    lb = cv2.resize(lb, outs[k].shape[1::-1], interpolation=cv2.INTER_NEAREST)
                    la = parser.parse(outs[k])
                    outs[k] = mirror_brow(tone_forehead(outs[k], lb, la), lb, la)
        finally:
            if parser is not None:
                parser.free()
        scores = self._score(images, outs, evaluators) if evaluators else {}
        return {k: (out, scores.get(k, {})) for k, out in outs.items()}

    def align(self, images: dict) -> dict:
        """{이름: 아무 사진} → {이름: FFHQ 정렬 1024 얼굴} (가장 큰 얼굴). 얼굴이 없으면 ValueError (메시지에 이름)."""
        detector = self.pipe._build("detector", self.pipe.cfg["detector"])
        try:
            out = {}
            for k, img in images.items():
                faces = self.pipe._detect(detector, img)
                if not faces:
                    raise ValueError(f"사진에서 얼굴을 찾지 못했습니다: {k}")
                out[k] = ffhq_align(img, faces[0].landmarks5, output_size=1024)
            return out
        finally:
            del detector
            self.pipe._free()

    def _plan(self, images, refs, parser, fit, scale):
        """대상마다 (넣을 참조 사진, 참조 번호, 배율, 대상/참조 얼굴 폭 비율). 얼굴을 못 재면 첫 참조를 그대로."""
        labs = [parser.parse(_to1024(r)) for r in refs]
        rm = [hair_fit.measure(lab) for lab in labs]
        bad = [bool(hair_fit.quality_fails(hair_fit.quality(lab, r))) for lab, r in zip(labs, refs)]
        plan = {}
        for k, img in images.items():
            tm = hair_fit.measure(parser.parse(_to1024(img)))
            i = hair_fit.pick(tm, rm, bad) if tm is not None and len(refs) > 1 else 0
            if tm is None or rm[i] is None:
                plan[k] = (refs[i], i, 1.0, None)
                continue
            ref_img, s = hair_fit.fit(_to1024(refs[i]), rm[i], tm, user_scale=scale, auto=fit)
            plan[k] = (ref_img, i, s, hair_fit.face_ratio(tm, rm[i]))
        return plan

    def _align_list(self, refs):
        """[(이미지, 정렬 필요 여부)] → 정렬된 이미지 목록."""
        todo = {f"머리 참조 {i + 1}": img for i, (img, needs) in enumerate(refs) if needs}
        done = self.align(todo) if todo else {}
        return [done.get(f"머리 참조 {i + 1}", img) for i, (img, _) in enumerate(refs)]

    def _align_refs(self, shape, color):
        todo = {n: r[0] for n, r in (("머리 모양 참조", shape), ("머리 색 참조", color)) if r and r[1]}
        done = self.align(todo) if todo else {}
        return done.get("머리 모양 참조", shape[0]), color and done.get("머리 색 참조", color[0])

    def _score(self, before: dict, after: dict, evaluators) -> dict:
        detector = self.pipe._build("detector", self.pipe.cfg["detector"])
        evs = {n: self.pipe._build("evaluator", {"name": n}) for n in evaluators}

        def embed(img):
            faces = self.pipe._detect(detector, img)
            if not faces:
                return None
            face112 = arcface_align(img, faces[0].landmarks5)
            return {n: ev.embed(face112) for n, ev in evs.items()}

        try:
            result = {}
            for k, out in after.items():
                a, b = embed(before[k]), embed(out)
                result[k] = {n: cosine(a[n], b[n]) for n in evs} if a and b else {}
            return result
        finally:
            del detector, evs
            self.pipe._free()

    def free(self):
        if self.styler is not None:
            self.styler.free()
            self.styler = None


def _to1024(img: np.ndarray) -> np.ndarray:
    return img if img.shape[:2] == (1024, 1024) else cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
