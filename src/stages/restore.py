"""3단계: 얼굴 복원. 정렬된 얼굴(RGB)을 받아 같은 크기로 돌려준다."""
import cv2
import numpy as np

from ..paths import add_third_party, weight
from ..registry import register
from ._stub import stub


@register("restorer", "none")
class NoRestore:
    def __init__(self, **_):
        pass

    def __call__(self, face_rgb: np.ndarray) -> np.ndarray:
        return face_rgb


@register("restorer", "codeformer")
class CodeFormerRestorer:
    """w: 0에 가까울수록 화질 우선, 1에 가까울수록 원본 신원 우선."""

    def __init__(self, device="cuda", w=0.5, **_):
        import torch
        add_third_party("CodeFormer")
        from basicsr.utils.registry import ARCH_REGISTRY

        self.torch = torch
        self.device = device
        self.w = w
        self.net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
            connect_list=["32", "64", "128", "256"]).to(device)
        ckpt = torch.load(weight("codeformer", "codeformer.pth"), map_location="cpu")["params_ema"]
        self.net.load_state_dict(ckpt)
        self.net.eval()

    def __call__(self, face_rgb: np.ndarray) -> np.ndarray:
        torch = self.torch
        h, w = face_rgb.shape[:2]
        x = cv2.resize(face_rgb, (512, 512), interpolation=cv2.INTER_AREA if h > 512 else cv2.INTER_CUBIC)
        x = torch.from_numpy(x).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.net(x, w=self.w, adain=True)[0]
        out = out.squeeze(0).clamp(-1, 1).add(1).div(2).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        if (h, w) != (512, 512):
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return out


stub("restorer", "gfpgan", "https://github.com/TencentARC/GFPGAN")
stub("restorer", "restoreformer", "https://github.com/wzhouxiff/RestoreFormerPlusPlus")
stub("restorer", "gpen", "https://github.com/yangxy/GPEN")
