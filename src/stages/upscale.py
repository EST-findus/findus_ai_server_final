"""업스케일. 검출이 안 될 만큼 작은 원본 사진을 키우거나, 최종 결과 화질을 올릴 때 쓴다."""
import numpy as np

from ..paths import add_third_party, weight
from ..registry import register
from ._stub import stub


@register("upscaler", "none")
class NoUpscale:
    scale = 1

    def __init__(self, **_):
        pass

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        return img_rgb


@register("upscaler", "realesrgan")
class RealESRGANUpscaler:
    """CodeFormer에 들어있는 basicsr의 RealESRGANer 사용 (x2plus)."""

    scale = 2

    def __init__(self, device="cuda", tile=400, **_):
        import torch
        add_third_party("CodeFormer")
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.realesrgan_utils import RealESRGANer

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        self.upsampler = RealESRGANer(
            scale=2, model_path=str(weight("realesrgan", "RealESRGAN_x2plus.pth")), model=model,
            tile=tile, tile_pad=40, pre_pad=0, half=(device == "cuda" and torch.cuda.is_available()),
        )

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        out, _ = self.upsampler.enhance(img_rgb[:, :, ::-1].copy(), outscale=self.scale)
        return out[:, :, ::-1].copy()


stub("upscaler", "swinir", "https://github.com/JingyunLiang/SwinIR")
stub("upscaler", "bsrgan", "https://github.com/cszn/BSRGAN")
