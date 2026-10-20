"""부모 조건 LoRA (학습형 부모 반영). SAM 본체는 고정하고 작은 모듈만 학습한다.

세 가지 모드 (체크포인트의 "mode"):
  latent   ParentAdapter. SAM 이 낸 W+ latent 에 층별 저랭크 잔차를 더한다 (3-A, 가볍고 빠름)
               Δw[l] = B_l · ( (A_l·w[l]) ⊙ g(c) + m · P_l·w_parent[l] )
           g(c) = 부모 조건으로 만든 rank 게이트, m = 부모가 있으면 1. B 는 0 으로 시작 → 처음엔 원본 SAM 과 같다.
  encoder  SAM 나이 인코더의 style 머리(map2style)에 조건부 LoRA (3-B). 게이트는 같은 g(c)
  recon    pSp 재구성 인코더에 비조건 LoRA (3-C, 도메인 적응: 한국인 얼굴·옛날 사진 → 인버전 손실 감소)

부모 조건 c = [부모 IR-SE50 임베딩 평균 · m, 부모 latent 층 평균 · m, 목표 나이/100, m]
"""
import math

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .paths import add_third_party, weight

N_STYLES, W_DIM, EMB_DIM = 18, 512, 512
COND_DIM = EMB_DIM + W_DIM + 2


# ---------------------------------------------------------------- 공통 도구

def to_tensor256(img_rgb: np.ndarray) -> torch.Tensor:
    """uint8 RGB → (3, 256, 256) [-1, 1]"""
    x = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA) if img_rgb.shape[0] != 256 else img_rgb
    return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float().div(127.5).sub(1)


@torch.no_grad()
def decode256(G, codes: torch.Tensor, chunk=2) -> torch.Tensor:
    """진짜 1024 출력을 256 으로 줄인 것 (기울기 없음, 메모리 때문에 chunk 장씩)."""
    return torch.cat([F.adaptive_avg_pool2d(G([codes[k:k + chunk]], input_is_latent=True, randomize_noise=False)[0], 256)
                      for k in range(0, len(codes), chunk)])


def decode_grad(G, codes: torch.Tensor) -> torch.Tensor:
    """1024 출력을 256 으로 줄인 것 (기울기 있음). bf16 이면 batch 1 에 ~2.9GB · 0.23초 → batch 1 + 기울기 누적으로 쓴다.
    (예전 straight-through 근사는 ID 손실 기울기 방향이 진짜와 코사인 0.55 밖에 안 맞아 뺐다)"""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        img = G([codes], input_is_latent=True, randomize_noise=False)[0]
    return F.adaptive_avg_pool2d(img.float(), 256)


class IDNet(nn.Module):
    """pSp ID 손실용 IR-SE50 (TreB1eN, MS1M). FFHQ 정렬 256 [-1,1] → L2 정규화 512.
    평가모델(ArcFace w600k_r50, AdaFace)과 다른 망이라 학습이 평가 점수를 직접 속이지 않는다."""

    def __init__(self):
        super().__init__()
        add_third_party("SAM")
        from models.encoders.model_irse import Backbone
        self.net = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode="ir_se")
        self.net.load_state_dict(torch.load(weight("ir_se50", "model_ir_se50.pth"), map_location="cpu"))
        self.net.eval().requires_grad_(False)

    def train(self, mode=True):  # BatchNorm/Dropout 은 항상 eval
        return super().train(False)

    def forward(self, x256: torch.Tensor) -> torch.Tensor:
        if x256.shape[-1] != 256:
            x256 = F.adaptive_avg_pool2d(x256, 256)
        return self.net(F.adaptive_avg_pool2d(x256[:, :, 35:223, 32:220], 112).contiguous())  # Flatten 이 view 를 씀


class HeldOut:
    """체크포인트 고르기용 ArcFace (w600k_r50). 학습 손실(IR-SE50)과 다른 망이라 '속이기'에 안 넘어간다.
    정렬은 pSp ID 손실과 같은 고정 크롭(FFHQ 256 → [35:223, 32:220] → 112)으로 근사."""

    def __init__(self):
        from .stages.evaluate import ArcFaceEvaluator
        self.ev = ArcFaceEvaluator()

    def __call__(self, x256: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x256[:, :, 35:223, 32:220].float(), size=(112, 112), mode="area")
        x = x.add(1).mul(127.5).clamp(0, 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        return F.normalize(torch.from_numpy(np.stack([self.ev.embed(f) for f in x])), dim=1)


def make_cond(par_emb, par_w, age, mask, latent_avg) -> torch.Tensor:
    """par_emb (N,512), par_w (N,18,512) 부모 평균 (없으면 아무 값), age (N,) 세, mask (N,) 1=부모 있음."""
    m = mask.view(-1, 1)
    pw = (par_w - latent_avg).mean(1)
    return torch.cat([par_emb * m, pw * m, age.view(-1, 1) / 100.0, m], 1)


def _gate_mlp(n_out, hidden=256):
    mlp = nn.Sequential(nn.Linear(COND_DIM, hidden), nn.SiLU(), nn.Linear(hidden, n_out))
    nn.init.zeros_(mlp[-1].weight)
    nn.init.zeros_(mlp[-1].bias)
    return mlp  # 출력 0 → 게이트 1 (조건 없는 LoRA 로 시작)


# ---------------------------------------------------------------- 3-A latent 어댑터

class ParentAdapter(nn.Module):
    def __init__(self, rank=8, layers=N_STYLES, dim=W_DIM, max_layer=N_STYLES):
        """max_layer: 이 층부터는 고치지 않는다. 8~17층(색·질감)을 열어 두면 IR-SE50 만 속이는
        부자연스러운 얼굴(하얀 피부, 자주색 입술)로 수렴했다 → 기본 학습은 0~7층(얼굴형·신원)만."""
        super().__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.randn(layers, rank, dim) / math.sqrt(dim))
        self.P = nn.Parameter(torch.randn(layers, rank, dim) / math.sqrt(dim))
        self.B = nn.Parameter(torch.zeros(layers, dim, rank))
        self.gate = _gate_mlp(layers * rank)
        self.register_buffer("latent_avg", torch.zeros(layers, dim))
        self.register_buffer("layer_mask", (torch.arange(layers) < max_layer).float().view(layers, 1))

    def forward(self, w, par_w, cond):
        """w (N,18,512) SAM 결과, par_w (N,18,512) 부모 평균, cond = make_cond(...). → w + Δw, Δw"""
        avg = self.latent_avg
        m = cond[:, -1].view(-1, 1, 1)
        g = 1 + self.gate(cond).view(len(w), -1, self.rank)
        h = torch.einsum("lrd,nld->nlr", self.A, w - avg) * g + torch.einsum("lrd,nld->nlr", self.P, par_w - avg) * m
        dw = torch.einsum("ldr,nlr->nld", self.B, h) * self.layer_mask
        return w + dw, dw


# ---------------------------------------------------------------- 3-B / 3-C 인코더 LoRA

class _LoRAPair(nn.Module):
    def __init__(self, a: nn.Module, b: nn.Module):
        super().__init__()
        self.A, self.B = a, b
        nn.init.zeros_(self.B.weight)


class LoRAWrap(nn.Module):
    """base(x) + B(A(x) ⊙ gate). base 는 고정. gate (N, rank) 는 EncoderLoRA.set_cond 가 넣는다."""

    def __init__(self, base: nn.Module, rank: int):
        super().__init__()
        self.base = base.requires_grad_(False)
        if isinstance(base, nn.Conv2d):
            self.lora = _LoRAPair(nn.Conv2d(base.in_channels, rank, base.kernel_size, base.stride, base.padding, bias=False),
                                  nn.Conv2d(rank, base.out_channels, 1, bias=False))
        else:  # EqualLinear (weight: out × in)
            out_dim, in_dim = base.weight.shape
            self.lora = _LoRAPair(nn.Linear(in_dim, rank, bias=False), nn.Linear(rank, out_dim, bias=False))
        self.gate = None
        self.enabled = True

    def forward(self, x):
        if not self.enabled:
            return self.base(x)
        h = self.lora.A(x)
        if self.gate is not None:
            h = h * (self.gate if h.ndim == 2 else self.gate[:, :, None, None])
        return self.base(x) + self.lora.B(h)


class EncoderLoRA(nn.Module):
    """GradualStyleEncoder 의 style 머리 18개(linear, 선택적으로 conv)에 LoRA 를 끼운다. cond=True 면 부모 게이트."""

    def __init__(self, encoder, rank=8, convs=False, cond=False):
        super().__init__()
        self.rank = rank
        wraps = []
        for style in encoder.styles:
            style.linear = LoRAWrap(style.linear, rank)
            wraps.append(style.linear)
            if convs:
                for j, m in enumerate(style.convs):
                    if isinstance(m, nn.Conv2d):
                        style.convs[j] = LoRAWrap(m, rank)
                        wraps.append(style.convs[j])
        self._wraps = wraps  # 리스트(등록 안 함): state_dict 에는 아래 loras 만 들어간다
        self.loras = nn.ModuleList(w.lora for w in wraps)
        self.gate = _gate_mlp(len(wraps) * rank) if cond else None

    def set_cond(self, cond):
        g = 1 + self.gate(cond).view(len(cond), len(self._wraps), self.rank)
        for i, w in enumerate(self._wraps):
            w.gate = g[:, i]

    def clear_cond(self):
        for w in self._wraps:
            w.gate = None

    def set_enabled(self, on: bool):
        """encoder 모드는 SAM 나이 인코더 안에 끼워져 있어서, 꺼 두지 않으면 '원본 SAM' 조건에도 LoRA 가 걸린다."""
        for w in self._wraps:
            w.enabled = on


def save_ckpt(path, mode: str, module: nn.Module, cfg: dict, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mode": mode, "cfg": cfg, "state": module.state_dict(), **extra}, path)


def load_into(ckpt: dict, net, device):
    """SAM 의 pSp(net) 에 체크포인트를 붙이고 모듈을 돌려준다."""
    mode, cfg = ckpt["mode"], ckpt["cfg"]
    if mode == "latent":
        m = ParentAdapter(rank=cfg["rank"], max_layer=cfg.get("max_layer", N_STYLES))
    elif mode == "encoder":
        m = EncoderLoRA(net.encoder, rank=cfg["rank"], convs=cfg.get("convs", False), cond=True)
    elif mode == "recon":
        m = EncoderLoRA(net.pretrained_encoder, rank=cfg["rank"], convs=cfg.get("convs", False), cond=False)
    else:
        raise ValueError(f"알 수 없는 모드: {mode}")
    state = dict(ckpt["state"])
    if mode == "latent" and "layer_mask" not in state:  # max_layer 이전 체크포인트 = 전 층
        state["layer_mask"] = m.layer_mask
    m.load_state_dict(state)
    return m.to(device).eval()
