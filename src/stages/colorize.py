"""흑백 컬러화. 흑백·세피아 사진(RGB)을 받아 같은 크기의 컬러 사진을 돌려준다.

밝기(Lab 의 L)는 입력 그대로 두고 색(a, b)만 모델이 채운다. 그래서 세피아처럼 색이 한 가지인 사진도
L 만 쓰므로 흑백과 똑같이 처리된다.
"""
import importlib
import sys

import cv2
import numpy as np

from ..paths import THIRD_PARTY, weight
from ..registry import register
from ._stub import stub


def color_spread(img_rgb: np.ndarray) -> float:
    """사진의 '색 퍼짐'. 흑백·세피아처럼 색이 한 가지뿐이면 0 에 가깝다.

    Lab 의 a, b 에서 각자의 중앙값(=사진 전체의 단색 톤)을 빼고 남은 색 세기의 평균.
    흑백은 0, 세피아·푸른 톤 흑백 사진은 1~3, 컬러 사진은 대부분 4 이상 (uint8 Lab 단위, docs/[08]).
    """
    small = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2Lab).astype(np.float32)
    a, b = lab[..., 1], lab[..., 2]
    return float(np.mean(np.hypot(a - np.median(a), b - np.median(b))))


def is_grayscale(img_rgb: np.ndarray, threshold: float) -> bool:
    return color_spread(img_rgb) < threshold


def _skin_mask(h: int, w: int) -> np.ndarray:
    """FFHQ 정렬 얼굴의 볼·코·이마 아래쪽 타원 (0~1, 가장자리 부드럽게). 눈·입이 일부 들어가도 중앙값을 쓰므로 괜찮다."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = ((xx / w - 0.5) / 0.17) ** 2 + ((yy / h - 0.56) / 0.20) ** 2
    return np.clip((1.3 - r) / 0.6, 0, 1)


def _face_mask(h: int, w: int) -> np.ndarray:
    """색을 옮길 범위: 얼굴과 목 (배경·머리는 덜 바꾼다)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = ((xx / w - 0.5) / 0.30) ** 2 + ((yy / h - 0.58) / 0.36) ** 2
    return np.clip((1.4 - r) / 0.8, 0, 1)


def _ab_stats(img_rgb: np.ndarray):
    lab = cv2.cvtColor(img_rgb.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    m = _skin_mask(*img_rgb.shape[:2]) > 0.5
    ab = lab[..., 1:][m]
    return lab, np.median(ab, 0), ab.std(0) + 1e-3


def match_skin_tone(img_rgb: np.ndarray, refs: list, strength: float = 0.7) -> np.ndarray:
    """컬러화한 얼굴의 피부색(Lab a·b)을 참조 컬러 사진(부모 등)의 피부색 쪽으로 옮긴다. 밝기 L 은 그대로.

    부모와 자녀의 피부색이 같지는 않으므로 평균만 strength 만큼 옮기고, 퍼짐(std)은 절반만 맞춘다.
    refs: FFHQ 정렬된 컬러 얼굴들. 없으면 입력을 그대로 돌려준다."""
    if not refs:
        return img_rgb
    lab, mu, sd = _ab_stats(img_rgb)
    stats = [_ab_stats(r)[1:] for r in refs]
    mu_r, sd_r = np.mean([s[0] for s in stats], 0), np.mean([s[1] for s in stats], 0)
    scale = 1 + 0.5 * strength * (sd_r / sd - 1)
    target = (lab[..., 1:] - mu) * scale + mu + strength * (mu_r - mu)
    k = _face_mask(*img_rgb.shape[:2])[..., None]
    lab[..., 1:] = k * target + (1 - k) * lab[..., 1:]
    out = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    return (out.clip(0, 1) * 255).round().astype(np.uint8)


@register("colorizer", "none")
class NoColorize:
    def __init__(self, **_):
        pass

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        return img_rgb


def _import_ddcolor():
    """third_party/DDColor 의 ddcolor 패키지만 가져온다.

    경로를 잠깐만 넣었다 빼서, DDColor 의 basicsr 폴더가 CodeFormer 의 basicsr 를 가리지 않게 한다.
    (ddcolor/model.py 는 patch_third_party.py 가 ddcolor/arch_utils 를 보게 바꿔 둔다)
    """
    if "ddcolor" in sys.modules:
        return sys.modules["ddcolor"]
    path = THIRD_PARTY / "DDColor"
    if not (path / "ddcolor" / "arch_utils").exists():
        raise FileNotFoundError(f"{path} 가 없거나 패치 전입니다. git clone 후 python scripts/patch_third_party.py")
    sys.path.insert(0, str(path))
    try:
        return importlib.import_module("ddcolor")
    finally:
        sys.path.remove(str(path))


@register("colorizer", "ddcolor")
class DDColorColorizer:
    """DDColor (ICCV 2023). model: ddcolor_modelscope(기본) | ddcolor_artistic | ddcolor_paper | ddcolor_paper_tiny"""

    SIZES = {"ddcolor_modelscope": "large", "ddcolor_artistic": "large", "ddcolor_paper": "large",
             "ddcolor_paper_tiny": "tiny"}

    def __init__(self, device="cuda", model="ddcolor_modelscope", input_size=512, **_):
        import torch
        if model not in self.SIZES:
            raise KeyError(f"DDColor model '{model}' 없음. 가능: {list(self.SIZES)}")
        ddcolor = _import_ddcolor()
        self.torch = torch
        self.device = device
        self.input_size = input_size
        # 가중치는 strict=False 로 읽히므로 크기(large/tiny)가 틀리면 조용히 엉뚱한 결과가 나온다 → 이름으로 고정
        self.net = ddcolor.build_ddcolor_model(
            ddcolor.DDColor, model_path=str(weight("ddcolor", f"{model}.pth")),
            input_size=input_size, model_size=self.SIZES[model], device=device)

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        """원본 DDColor ColorizationPipeline 과 같지만, ab 를 늘릴 때 nearest 대신 bilinear 를 쓴다."""
        torch = self.torch
        h, w = img_rgb.shape[:2]
        img = img_rgb.astype(np.float32) / 255
        orig_l = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[..., :1]

        small_l = cv2.cvtColor(cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA),
                               cv2.COLOR_RGB2Lab)[..., :1]
        gray = cv2.cvtColor(np.concatenate([small_l, np.zeros_like(small_l), np.zeros_like(small_l)], -1),
                            cv2.COLOR_LAB2RGB)
        x = torch.from_numpy(gray.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            ab = self.net(x)
            ab = torch.nn.functional.interpolate(ab, size=(h, w), mode="bilinear", align_corners=False)
        ab = ab[0].float().cpu().numpy().transpose(1, 2, 0)
        out = cv2.cvtColor(np.concatenate([orig_l, ab], -1), cv2.COLOR_LAB2RGB)
        return (out.clip(0, 1) * 255).round().astype(np.uint8)


@register("colorizer", "codeformer_color")
class CodeFormerColorizer:
    """CodeFormer 얼굴 컬러화 모드 (얼굴 전용, 정렬 512 입력). 복원도 함께 일어나 얼굴 모양이 바뀔 수 있어서
    모델 결과에서 색(a, b)만 가져오고 밝기 L 은 입력 그대로 둔다 (DDColor 와 같은 원칙). 라이선스: S-Lab 비상업."""

    def __init__(self, device="cuda", **_):
        import torch
        from ..paths import add_third_party
        add_third_party("CodeFormer")
        from basicsr.utils.registry import ARCH_REGISTRY

        self.torch = torch
        self.device = device
        self.net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=["32", "64", "128"]).to(device)
        self.net.load_state_dict(torch.load(weight("codeformer", "codeformer_colorization.pth"),
                                            map_location="cpu")["params_ema"])
        self.net.eval()

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        torch = self.torch
        h, w = img_rgb.shape[:2]
        img = img_rgb.astype(np.float32) / 255
        orig_l = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[..., :1]
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        x = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA if h > 512 else cv2.INTER_CUBIC)
        x = torch.from_numpy(x.transpose(2, 0, 1)).sub(0.5).div(0.5).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.net(x, w=0, adain=True)[0]
        out = out[0].clamp(-1, 1).add(1).div(2).permute(1, 2, 0).cpu().numpy()
        ab = cv2.resize(cv2.cvtColor(out, cv2.COLOR_RGB2Lab)[..., 1:], (w, h), interpolation=cv2.INTER_LINEAR)
        res = cv2.cvtColor(np.concatenate([orig_l, ab], -1), cv2.COLOR_LAB2RGB)
        return (res.clip(0, 1) * 255).round().astype(np.uint8)


stub("colorizer", "deoldify", "https://github.com/jantic/DeOldify")
