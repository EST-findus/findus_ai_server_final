"""third_party 저장소를 Windows(MSVC 없음)에서 돌아가게 패치한다. 여러 번 실행해도 안전하다.

1. SAM의 StyleGAN2 CUDA 커스텀 연산(fused_act, upfirdn2d)을 순수 PyTorch 구현으로 교체
   - 원본은 *.py.cuda_bak 으로 보관
   - HairFastGAN·GlassesGAN(받아 둔 경우)도 같은 연산을 쓰므로 함께 교체
2. CodeFormer의 basicsr/version.py 생성 (원래 setup.py develop 이 만들어주는 파일)
3. DDColor(받아 둔 경우)가 자기 basicsr 폴더 대신 ddcolor/arch_utils 를 쓰게 바꾼다
   - CodeFormer 의 basicsr 와 이름이 같아서, 그대로 두면 한쪽이 가려진다
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DDCOLOR = ROOT / "third_party" / "DDColor"
SAM_OP = ROOT / "third_party" / "SAM" / "models" / "stylegan2" / "op"
HAIRFAST_OP = ROOT / "third_party" / "HairFastGAN" / "models" / "stylegan2" / "op"  # e4e·FS 인코더도 이 op 를 import
GLASSESGAN_OP = ROOT / "third_party" / "GlassesGAN" / "encoder4editing" / "models" / "stylegan2" / "op"  # 안경 (docs/[05])
CF_BASICSR = ROOT / "third_party" / "CodeFormer" / "basicsr"

FUSED_ACT = '''"""Pure PyTorch FusedLeakyReLU (patched by ai_lab/scripts/patch_third_party.py)."""
import torch
from torch import nn
from torch.nn import functional as F


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    rest_dim = [1] * (input.ndim - bias.ndim - 1)
    return F.leaky_relu(input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope) * scale
'''

UPFIRDN2D = '''"""Pure PyTorch upfirdn2d (patched by ai_lab/scripts/patch_third_party.py)."""
import torch
from torch.nn import functional as F


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0), :]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
                      in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
    return out.view(-1, channel, out_h, out_w)
'''


def patch_file(path: Path, content: str):
    backup = path.with_suffix(".py.cuda_bak")
    if not backup.exists():
        path.rename(backup)
    path.write_text(content, encoding="utf-8")
    print(f"patched {path.relative_to(ROOT)}")


def main():
    patch_file(SAM_OP / "fused_act.py", FUSED_ACT)
    patch_file(SAM_OP / "upfirdn2d.py", UPFIRDN2D)
    if HAIRFAST_OP.exists():
        patch_file(HAIRFAST_OP / "fused_act.py", FUSED_ACT)
        patch_file(HAIRFAST_OP / "upfirdn2d.py", UPFIRDN2D)
    if GLASSESGAN_OP.exists():
        patch_file(GLASSESGAN_OP / "fused_act.py", FUSED_ACT)
        patch_file(GLASSESGAN_OP / "upfirdn2d.py", UPFIRDN2D)

    version = (CF_BASICSR / "VERSION").read_text().strip()
    (CF_BASICSR / "version.py").write_text(
        f"__version__ = '{version}'\n__gitsha__ = 'unknown'\n"
        f"version_info = ({', '.join(version.split('.'))})\n",
        encoding="utf-8",
    )
    print("wrote third_party/CodeFormer/basicsr/version.py")

    if DDCOLOR.exists():
        dst = DDCOLOR / "ddcolor" / "arch_utils"
        if not dst.exists():
            shutil.copytree(DDCOLOR / "basicsr" / "archs" / "ddcolor_arch_utils", dst,
                            ignore=shutil.ignore_patterns("__pycache__"))
        model_py = DDCOLOR / "ddcolor" / "model.py"
        model_py.write_text(model_py.read_text(encoding="utf-8").replace(
            "from basicsr.archs.ddcolor_arch_utils.", "from .arch_utils."), encoding="utf-8")
        print("patched third_party/DDColor/ddcolor (basicsr → ddcolor/arch_utils)")


if __name__ == "__main__":
    main()
