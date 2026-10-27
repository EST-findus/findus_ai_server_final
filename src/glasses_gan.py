"""안경 씌우기 (GlassesGAN, CVPR 2023): 생성기가 안경을 그리고 테만 원본에 섞는다 (docs/[05]).

  1) e4e 로 얼굴 → W+ latent (18x512). 생성기는 SAM 과 같은 FFHQ StyleGAN2 (가중치가 완전히 같다)
  2) + 평균 안경 방향 × 세기. 세기는 테 면적이 목표(0.020)에 가장 가까운 값을 5개 중에서 고른다
  3) + 모양 축(PCA) 6개: 크기·높이·사각·둥근·캣아이·두께. 안경 종류마다 프리셋 값 (configs/glasses_presets.yaml)
  4) 생성 → DeepLab 으로 테(1)·오른 렌즈(2)·왼 렌즈(3) 분할
  5) 테 색을 Lab 에서 목표 색(검정·진회색·은색)으로 고정 (명암은 남김)
  6) 테를 또렷하게 붙인다 (composite): 테 = DeepLab 테 + 안경 없는 생성 얼굴보다 확 어두워진 곳(놓친 가는 테).
     테 둘레 그림자는 밝기 비율만 원본에 곱한다. 원본 유지: 렌즈 가운데와 안경 밖은 입력 그대로

원본 GlassesGAN 의 Colab 래퍼(체크포인트 복사, wget, 임시 파일·하위 프로세스로 분할)는 쓰지 않고 모델만 직접 불러 같은 순서로 돈다.
SAM 과 최상위 패키지 이름(models, utils …)이 겹쳐서 불러올 때만 잠깐 바꿔 끼운다 (src/hairstyle.HairStyler 와 같은 방식).

    gg = GlassesGAN("cuda")
    out, info = gg(face_1024_rgb, preset, size=0, thickness=0)
    gg.free()
"""
import os
import sys
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import yaml

from .paths import THIRD_PARTY, WEIGHTS

GG_DIR = THIRD_PARTY / "GlassesGAN"
W = WEIGHTS / "glassesgan"
FILES = {"e4e": "e4e_ffhq_encode.pt", "deeplab": "deeplab_epoch_19.pth",
         "pca": "fitted_pca_celebhq_dataset_results.joblib", "avg": "aveglassesdiff_celebhq_dataset_results_RG.npy"}
PRESETS = Path(__file__).resolve().parents[1] / "configs" / "glasses_presets.yaml"
AXES = ("size", "height", "square", "round", "cateye", "thick")   # PCA 축 0~5 (노트북 슬라이더 순서)
BIASES = (1.0, 1.25, 1.5, 1.75, 2.0)   # 평균 안경 세기 후보. 원본은 0.5~1.5 를 20번 생성했지만 사진 3장 모두 끝값(1.45~1.5)을 골랐다
TARGET_AREA = 0.020                    # 원본 ideal_avg_glasses_frame_area
NEAR = 18                              # 안경(테+렌즈)에서 이만큼(px) 안만 바꾼다. 밖은 원본 그대로
GROW = 4                               # 렌즈 밖에서 놓친 테를 찾는 폭 (DeepLab 테에서 px)
LINE = 20                              # 렌즈 안 놓친 테: 주변보다·안경 없는 얼굴보다 이만큼(Lab 밝기) 어두운 가는 선
RIM_DARK = 10                          # 렌즈 안에서 '확실히 어두워진 곳'을 테로 보는 폭 (더 넓히면 머리 그늘 덩어리가 잡혔다)
RIM_IN = 16                            # 렌즈 안에서 놓친 테를 찾는 폭 (렌즈 경계에서 px). 더 안쪽 어두운 덩어리는 머리 그늘 등
CORE = 0.35                            # 렌즈 가운데: 렌즈 안 거리 최댓값의 이 비율보다 안쪽은 원본 그대로 (눈)
DARKER = 45                            # recon 보다 Lab 밝기(0~255)가 이만큼 어두워지고 DeepLab 테와 이어지면 테로 본다
SHADE = (0.82, 1.03)                   # 테 둘레 그림자·반사 밝기 비율 범위 (더 넓으면 테 아래 회색 띠)
SHADE_R = 5                            # 그림자는 테에서 이만큼(px) 안에만. 넓으면 테 아래에 회색 띠가 생겼다
FINE = 4.0                             # 요청의 size·thickness(-1~1) 1 당 PCA 축 이동량
_SHARED = ("models", "utils", "configs", "criteria", "datasets", "editings", "options", "scripts", "training")
_MODULES = {}


def available() -> bool:
    return GG_DIR.is_dir() and all((W / f).exists() for f in FILES.values())


def presets() -> dict:
    return yaml.safe_load(PRESETS.read_text(encoding="utf-8"))


def preset(key: str):
    """configs/glasses_presets.yaml 에서 key 의 안경. 없으면 None."""
    return next((p for p in presets()["styles"] if p["key"] == key), None)


@contextmanager
def _isolated():
    """GlassesGAN 의 encoder4editing 모듈을 잠깐 바꿔 끼운다 (SAM 의 models 와 이름이 같다)."""
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.split(".")[0] in _SHARED}
    path = sys.path[:]
    sys.modules.update(_MODULES)
    sys.path.insert(0, str(GG_DIR / "encoder4editing"))
    try:
        yield
    finally:
        _MODULES.update({k: sys.modules.pop(k) for k in list(sys.modules) if k.split(".")[0] in _SHARED})
        sys.modules.update(saved)
        sys.path[:] = path


def recolor_frame(img: np.ndarray, frame: np.ndarray, color: dict) -> np.ndarray:
    """테 영역(0~1 가중치)의 색 맞추기: 채도(a·b)는 sat 배로 줄이고, L 이 있으면 밝기 평균만 그쪽으로 옮긴다 (명암은 그대로).
    밝은 색(은색 등)으로 통째로 칠하면 만화처럼 납작해져서 쓰지 않는다 (2026-10-27 사진 3장에서 확인)."""
    w = frame > 0.5
    if w.sum() < 50:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    new = lab.copy()
    if color.get("L") is not None:  # 밝기 평균을 목표로, 명암 폭은 contrast 배 (검정 테는 줄여야 밝게 그려진 아래 테가 끊겨 보이지 않는다)
        mean = float(lab[:, :, 0][w].mean())
        new[:, :, 0] = color["L"] * 255 / 100 + (lab[:, :, 0] - mean) * float(color.get("contrast", 1.0))
    sat = float(color.get("sat", 0.0))
    new[:, :, 1:] = 128 + (lab[:, :, 1:] - 128) * sat
    rgb = cv2.cvtColor(np.clip(new, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    f = frame[..., None]
    return np.clip(img * (1 - f) + rgb * f, 0, 255).round().astype(np.uint8)


def composite(face: np.ndarray, edit: np.ndarray, recon: np.ndarray, seg: np.ndarray, color: dict) -> tuple:
    """원본 + 생성 결과 → (안경 쓴 얼굴, 테 마스크, 렌즈 가운데 마스크).
    - 테: DeepLab 테 ∪ (안경 근처에서 recon 보다 DARKER 이상 어두워진 곳). 가는 테를 렌즈로 잘못 본 곳도 테로 잡혀 끊기지 않는다
    - 테는 또렷하게 붙인다 (경계 1px 만 흐림). 넓게 흐려 섞으면 테 끝이 반투명하게 번졌다
    - 테 둘레 그림자·반사는 픽셀을 섞지 않고 밝기 비율(edit/recon)만 원본 피부에 곱한다 (색은 원본)
    - 원본 유지: 렌즈 가운데(core)와 안경에서 NEAR px 밖은 입력 그대로"""
    Le = cv2.cvtColor(edit, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
    Lr = cv2.cvtColor(recon, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
    ell = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))  # noqa: E731
    glasses = (seg > 0).astype(np.uint8)
    near = cv2.dilate(glasses, ell(NEAR)) > 0
    lens = ((seg == 2) | (seg == 3)).astype(np.uint8)
    dist = cv2.distanceTransform(lens, cv2.DIST_L2, 5)
    core = dist > CORE * max(float(dist.max()), 1.0)       # 렌즈 가운데 (눈): 테가 올 수 없는 곳
    dl_frame = (seg == 1).astype(np.uint8)
    # 놓친 테: 안경 근처(렌즈 가운데 빼고)에서 확실히 어두워졌고 DeepLab 테와 이어진 덩어리만.
    # DeepLab 이 아래 테 가운데를 렌즈로 보는 일이 있어서(밝은 얼굴) 렌즈 영역도 찾고, 눈·그늘처럼 테와 안 이어진 것은 뺀다
    # 렌즈 밖: DeepLab 테 바로 옆(GROW)에서 확실히 어두워진 곳 (넓히면 눈썹이 테로 칠해졌다)
    out_lens = (lens == 0) & (cv2.dilate(dl_frame, ell(GROW)) > 0) & (Lr - Le > DARKER) & (Le < 0.6 * Lr)
    # 렌즈 안: 가장자리 RIM_DARK px 는 확실히 어두워진 곳, RIM_IN px 까지는 가는 어두운 선(black-hat)이면서 안경 없는 얼굴엔
    # 없던 것. 덩어리(머리 그늘)·눈 선은 빠지고, 금속 테처럼 덜 어두운 테도 선이라서 잡힌다
    dark = (Lr - Le > DARKER) & (Le < 0.6 * Lr)
    blackhat = cv2.morphologyEx(Le, cv2.MORPH_BLACKHAT, ell(5))
    in_lens = (lens > 0) & (((dist <= RIM_DARK) & dark) | ((dist <= RIM_IN) & (blackhat > LINE) & (Lr - Le > LINE)))
    cand = ((out_lens | in_lens) & ~core).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, ell(1))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand | cv2.dilate(dl_frame, ell(1)))
    touch = set(np.unique(lab[dl_frame > 0])) - {0}
    missed = np.isin(lab, list(touch)) & (cand > 0)
    frame = (dl_frame.astype(bool) | missed).astype(np.uint8)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, ell(2))
    alpha = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), 0.8)
    alpha[core] = 0

    lab_f = cv2.cvtColor(face, cv2.COLOR_RGB2LAB).astype(np.float32)
    # 그림자: 테 밖·렌즈 가운데 밖에서 밝기 비율만. 경계는 부드럽게
    ratio = cv2.GaussianBlur(np.clip((Le + 8) / (Lr + 8), SHADE[0], SHADE[1]), (0, 0), 1.5)
    ring = (cv2.dilate(frame, ell(SHADE_R)) > 0) & (lens == 0)
    shade_w = cv2.GaussianBlur(ring.astype(np.float32), (0, 0), 3) * (1 - alpha)
    shade_w[lens > 0] = 0   # 렌즈 안은 그림자도 넣지 않는다 (눈 주변이 얼룩처럼 어두워졌다)
    shaded = lab_f.copy()
    shaded[:, :, 0] = np.clip(lab_f[:, :, 0] * (1 + (ratio - 1) * shade_w), 0, 255)
    base = cv2.cvtColor(shaded.round().astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    fr = recolor_frame(edit, alpha, color)
    # 테 안의 밝은 띠(렌즈 단면 비침)는 회색이면 피부 옆에서 푸르게 보여서 색만 원본 피부색으로 (지우면 아래 테가 끊긴다)
    lab_fr = cv2.cvtColor(fr, cv2.COLOR_RGB2LAB).astype(np.float32)
    w = np.clip((140 - lab_fr[:, :, :1]) / 80, 0, 1)   # 색을 맞춘 뒤 밝기로 본다
    lab_fr[:, :, 1:] = w * lab_fr[:, :, 1:] + (1 - w) * lab_f[:, :, 1:]
    fr = cv2.cvtColor(np.clip(lab_fr, 0, 255).round().astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    out = np.clip(base * (1 - alpha[..., None]) + fr * alpha[..., None], 0, 255).round().astype(np.uint8)
    keep = core | ~near
    out[keep] = face[keep]
    return out, frame > 0, core


class GlassesGAN:
    def __init__(self, device: str = "cuda"):
        if not available():
            raise FileNotFoundError(f"GlassesGAN 이 없습니다: {GG_DIR} / {W} → git clone pleshro/GlassesGAN_release "
                                    "third_party/GlassesGAN, python scripts/download_assets.py, python scripts/patch_third_party.py")
        import joblib
        import torch
        import torchvision
        self.torch, self.device = torch, device
        ck = torch.load(W / FILES["e4e"], map_location="cpu")
        opts = dict(ck["opts"], checkpoint_path=str(W / FILES["e4e"]), device=device)
        del ck
        with _isolated():
            from models.psp import pSp
            self.net = pSp(Namespace(**opts)).eval().to(device)
        dl = torchvision.models.segmentation.deeplabv3_resnet101(weights=None, weights_backbone=None, num_classes=4, aux_loss=None)
        dl.load_state_dict(torch.load(W / FILES["deeplab"], map_location="cpu")["model_state_dict"])
        self.dl = dl.eval().to(device)
        pca = joblib.load(W / FILES["pca"])
        # 원본은 축마다 pca.inverse_transform(one_hot) 를 더한다 = 축 × 값 + mean_. 받은 파일의 mean_ 은 0 이라 축만 쓴다
        assert float(np.abs(pca.mean_).max()) < 1e-6
        self.axes = pca.components_[:len(AXES)].astype(np.float32)
        self.avg = np.load(W / FILES["avg"]).astype(np.float32)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # ---------------------------------------------------------------- 모델

    def encode(self, rgb: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(x).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(self.device)
        with torch.no_grad():
            return (self.net.encoder(x) + self.net.latent_avg.repeat(1, 1, 1)).cpu().numpy()

    def generate(self, lat: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            img, _ = self.net.decoder([torch.from_numpy(lat.astype(np.float32)).to(self.device)],
                                      input_is_latent=True, randomize_noise=False)
        return ((img[0].clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).round().byte().cpu().numpy()

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        """0 배경, 1 테, 2 오른 렌즈, 3 왼 렌즈 (입력 크기)."""
        torch = self.torch
        x = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(x).permute(2, 0, 1)[None].float().div(255).to(self.device)
        with torch.no_grad():
            lab = self.dl((x - self.mean) / self.std)["out"].argmax(1)[0].byte().cpu().numpy()
        return cv2.resize(lab, rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST)

    def shape_offset(self, values: dict) -> np.ndarray:
        v = np.array([float(values.get(a, 0)) for a in AXES], np.float32)
        return (v @ self.axes).reshape(1, 18, 512)

    # ---------------------------------------------------------------- 안경

    def __call__(self, face: np.ndarray, preset: dict, size: float = 0.0, thickness: float = 0.0,
                 latent: np.ndarray = None) -> tuple:
        """FFHQ 정렬 1024 얼굴(RGB) + 프리셋 → (안경 쓴 얼굴, 정보)."""
        if face.shape[:2] != (1024, 1024):
            face = cv2.resize(face, (1024, 1024), interpolation=cv2.INTER_AREA)
        lat = self.encode(face) if latent is None else latent
        if preset.get("bias"):
            bias, area = float(preset["bias"]), None
        else:  # 평균 안경 세기: 테 면적이 목표에 가장 가까운 것
            areas = [float((self.segment(self.generate(lat + self.avg * b)) == 1).mean()) for b in BIASES]
            i = int(np.argmin(np.abs(np.array(areas) - TARGET_AREA)))
            bias, area = BIASES[i], areas[i]
        shape = dict(preset.get("shape") or {})
        shape["size"] = shape.get("size", 0) + FINE * size
        shape["thick"] = shape.get("thick", 0) + FINE * thickness
        edit = self.generate(lat + self.avg * bias + self.shape_offset(shape))
        recon = self.generate(lat)   # 같은 latent 의 안경 없는 얼굴: 안경 때문에 어두워진 곳 = 테 (DeepLab 이 놓친 가는 테까지)
        seg = self.segment(edit)
        if (seg == 1).mean() < 0.002:
            raise ValueError("안경이 생성되지 않았습니다 (테 면적이 너무 작음). 크기·두께를 키워 보세요")
        out, frame, core = composite(face, edit, recon, seg, preset.get("color") or {})
        changed = np.any(out != face, axis=2)
        return out, {"bias": bias, "frame_area": round(float(frame.mean()), 4),
                     "auto_area": None if area is None else round(area, 4),
                     "changed_ratio": round(float(changed.mean()), 4),
                     "lens_changed": round(float(changed[core].mean()), 4) if core.any() else None}

    def preview(self, preset: dict) -> np.ndarray:
        """카드용 미리보기: 생성기의 평균 얼굴(실존 인물 아님)에 씌운 눈 부분 (480x240 RGB)."""
        avg = self.net.latent_avg.repeat(1, 1, 1).cpu().numpy()
        img, _ = self(self.generate(avg), preset, latent=avg)
        return cv2.resize(img[330:700, 140:884], (480, 240), interpolation=cv2.INTER_AREA)

    def free(self):
        import gc
        self.net = self.dl = None
        gc.collect()
        self.torch.cuda.empty_cache()
