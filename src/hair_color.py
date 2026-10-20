"""머리 색 정확히 지정하기: 얼굴 분할(BiSeNet, CelebAMask-HQ 19부위)로 머리 영역을 찾아 Lab 색공간에서 다시 칠한다.

    hm = HairMask()
    mask = hm(face_rgb)                         # 0~1, 머리만 (가장자리 부드럽게)
    out = recolor(face_rgb, mask, "#6B5A4E")    # 애쉬 브라운
    hm.free()

HairFastGAN 은 색을 '참조 사진'으로만 받아서 원하는 색이 정확히 나오지 않는다. 그래서
  (1) 머리 모양은 HairFastGAN 으로 바꾸고, (2) 색은 그 결과의 머리 영역을 목표 색으로 칠한다.
칠할 때 원래 머리의 명암(결·윤기)은 남기고 평균 밝기와 색만 목표로 옮긴다 → 염색한 것처럼 보인다.
가중치는 HairFastGAN 이 받아 둔 third_party/HairFastGAN/pretrained_models/BiSeNet/face_parsing_79999_iter.pth.
"""
import importlib.util
import sys
import types

import cv2
import numpy as np

from .paths import THIRD_PARTY

HAIR, HAT = 17, 18
FP_DIR = THIRD_PARTY / "HairFastGAN" / "models" / "face_parsing"
FP_WEIGHT = THIRD_PARTY / "HairFastGAN" / "pretrained_models" / "BiSeNet" / "face_parsing_79999_iter.pth"


def _load_bisenet():
    """HairFast 의 face_parsing 코드를 'hf_face_parsing' 라는 따로 된 이름으로 읽는다 (SAM 의 models 패키지와 충돌 방지)."""
    name = "hf_face_parsing"
    if name + ".model" in sys.modules:
        return sys.modules[name + ".model"].BiSeNet
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(FP_DIR)]
    sys.modules[name] = pkg
    for sub in ("resnet", "model"):
        spec = importlib.util.spec_from_file_location(f"{name}.{sub}", FP_DIR / f"{sub}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"{name}.{sub}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name + ".model"].BiSeNet


class HairMask:
    def __init__(self, device="cuda"):
        import torch
        if not FP_WEIGHT.exists():
            raise FileNotFoundError(f"{FP_WEIGHT} 가 없습니다 → python scripts/try_hairfast.py --download")
        self.torch, self.device = torch, device
        self.net = _load_bisenet()(n_classes=19).to(device).eval()
        self.net.load_state_dict(torch.load(FP_WEIGHT, map_location="cpu"))
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _prob(self, img_rgb: np.ndarray):
        """19부위 확률 (19, 512, 512) torch 텐서."""
        torch = self.torch
        h = img_rgb.shape[0]
        x = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA if h > 512 else cv2.INTER_CUBIC)
        x = torch.from_numpy(x).permute(2, 0, 1)[None].float().div(255).to(self.device)
        with torch.no_grad():
            return self.net((x - self.mean) / self.std)[0].softmax(1)[0]

    def parse(self, img_rgb: np.ndarray) -> np.ndarray:
        """FFHQ 정렬 얼굴 → 부위 라벨 지도 (H, W) uint8. CelebAMask-HQ 순서:
        0 배경, 1 피부, 2·3 눈썹, 4·5 눈, 6 안경, 7·8 귀, 9 귀걸이, 10 코, 11 입, 12·13 입술, 14 목, 15 목걸이, 16 옷, 17 머리, 18 모자."""
        h, w = img_rgb.shape[:2]
        lab = self._prob(img_rgb).argmax(0).byte().cpu().numpy()
        return cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)

    def __call__(self, img_rgb: np.ndarray, feather: int = 7) -> np.ndarray:
        """FFHQ 정렬 얼굴 → 머리 확률 마스크 (H, W) 0~1."""
        h, w = img_rgb.shape[:2]
        hair = self._prob(img_rgb)[HAIR].cpu().numpy().astype(np.float32)
        hair = cv2.resize(hair, (w, h), interpolation=cv2.INTER_LINEAR)
        k = max(1, int(feather * w / 512)) | 1
        return cv2.GaussianBlur(hair, (k, k), 0).clip(0, 1)

    def free(self):
        self.net = None
        self.torch.cuda.empty_cache()


def hex_to_lab(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    rgb = np.array([[[int(h[i:i + 2], 16) for i in (0, 2, 4)]]], np.float32) / 255
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[0, 0]  # L 0~100, a·b 약 -127~127


def recolor(img_rgb: np.ndarray, mask: np.ndarray, hex_color: str, contrast: float = 1.0,
            texture: float = 0.35) -> np.ndarray:
    """머리 영역을 목표 색으로. 결과 머리의 평균 Lab 이 목표 색이 되게 옮기고 명암 결은 남긴다.
    contrast: 원래 밝기 결을 얼마나 살릴지 (밝은 색으로 바꿀 때 1.2 정도면 덜 평평해 보인다).
    texture: 원래 색(a·b) 얼룩을 얼마나 남길지 (0 = 완전히 단색, 1 = 원래 얼룩 그대로)."""
    lab = cv2.cvtColor(img_rgb.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    m = mask > 0.8  # 확실한 머리 (통계·보정 기준)
    if m.sum() < 50:  # 머리가 거의 안 보이면 (모자·민머리) 그대로
        return img_rgb
    tgt = hex_to_lab(hex_color)
    L, ab = lab[..., 0], lab[..., 1:]
    mL, mab = L[m].mean(), ab[m].mean(0)
    # 밝은 목표로 갈수록 원래 결의 대비를 조금 키워야 평평해 보이지 않는다
    k = contrast * (1 + max(0.0, tgt[0] - mL) / 200)
    new = np.empty_like(lab)
    new[..., 0] = tgt[0] + k * (L - mL)
    new[..., 1:] = tgt[1:] + texture * (ab - mab)
    a = mask[..., None]
    # 가장자리 섞임·잘림 때문에 평균이 목표에서 벗어난 만큼 두 번 다시 맞춘다
    for _ in range(2):
        out = a * new + (1 - a) * lab
        out[..., 0] = out[..., 0].clip(0, 100)
        new += (tgt - out[m].mean(0)) / mask[m].mean()
    out = a * new + (1 - a) * lab
    out[..., 0] = out[..., 0].clip(0, 100)
    res = cv2.cvtColor(out, cv2.COLOR_Lab2RGB)
    return (res.clip(0, 1) * 255).round().astype(np.uint8)


def hair_delta_e(img_rgb: np.ndarray, mask: np.ndarray, hex_color: str) -> float:
    """결과 머리 평균색과 목표 색의 차이 (CIE76 ΔE, 2 이하면 눈으로 구분 어려움)."""
    lab = cv2.cvtColor(img_rgb.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    m = mask > 0.8
    return float(np.linalg.norm(lab[m].mean(0) - hex_to_lab(hex_color))) if m.sum() >= 50 else float("nan")
