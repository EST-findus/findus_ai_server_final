"""결과 얼굴의 추정 나이 (MiVOLO 얼굴 전용 volo_d1, IMDB-cleaned). 평가용이라 결과를 걸러내지 않고 보여 주기만 한다.

SAM 학습 손실의 DEX 나이 분류기와 다른 망이다 (학습용과 평가용을 섞지 않는다).
CPU 로 돌아서 GPU 차례를 기다리지 않는다. 버튼을 누른 사진만 추정한다 (GET /v1/results/{id}/images/{key}/age).

    python -m app.age_eval outputs/<result_id>/age30.png     # 한 장 추정 (자기검사)
"""
import threading
from typing import Optional

import cv2
import numpy as np

from src.paths import THIRD_PARTY, WEIGHTS, add_third_party

CKPT = WEIGHTS / "mivolo" / "mivolo_face_imdb.pth.tar"
# FFHQ 정렬 1024 결과의 RetinaFace 얼굴 박스 평균 (나이 변환 결과 30장, 표준편차 9~19px). 검출기를 안 돌려도 된다
FACE_BOX = (243, 235, 771, 920)
_MEAN = np.array([0.485, 0.456, 0.406])  # ImageNet (timm IMAGENET_DEFAULT_MEAN/STD 와 같은 값)
_STD = np.array([0.229, 0.224, 0.225])

_lock = threading.Lock()
_model = None  # (model, meta) 한 번 올려서 재사용


def missing() -> Optional[str]:
    """준비가 안 됐으면 이유, 다 있으면 None."""
    if not (THIRD_PARTY / "MiVOLO").exists():
        return "third_party/MiVOLO 가 없습니다 (git clone --depth 1 https://github.com/WildChlamydia/MiVOLO third_party/MiVOLO)"
    if not CKPT.exists():
        return "가중치가 없습니다: weights/mivolo/mivolo_face_imdb.pth.tar (python scripts/download_assets.py)"
    try:
        import timm  # noqa: F401
    except ImportError:
        return "timm 이 없습니다 (pip install timm==0.8.13.dev0 --no-deps)"
    return None


def _load():
    global _model
    if _model is None:
        import torch
        add_third_party("MiVOLO")
        from mivolo.model.create_timm_model import create_model
        state = torch.load(CKPT, map_location="cpu")
        size = state["state_dict"]["pos_embed"].shape[1] * 16
        model = create_model(f"mivolo_d1_{size}", num_classes=1 if state["no_gender"] else 3, in_chans=3,
                             checkpoint_path=str(CKPT), filter_keys=["fds."]).eval()
        _model = model, {k: state[k] for k in ("min_age", "max_age", "avg_age", "no_gender")}, size
    return _model


def _letterbox(img: np.ndarray, size: int) -> np.ndarray:
    """비율 유지 축소 + 검은 여백 (MiVOLO data/misc.py class_letterbox 와 같게)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    return cv2.copyMakeBorder(img, int(round(dh - 0.1)), int(round(dh + 0.1)), int(round(dw - 0.1)), int(round(dw + 0.1)),
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def estimate(img_rgb: np.ndarray) -> Optional[float]:
    """FFHQ 정렬 1024 얼굴(RGB) → 추정 나이. 준비가 안 됐으면 None."""
    if missing():
        return None
    import torch
    with _lock:
        model, meta, size = _load()
        s = img_rgb.shape[0] / 1024  # 1024 가 아닌 정렬 얼굴도 같은 비율로
        x1, y1, x2, y2 = (int(round(v * s)) for v in FACE_BOX)
        x = (_letterbox(img_rgb[y1:y2, x1:x2], size) / 255.0 - _MEAN) / _STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32))[None]
        with torch.no_grad():
            out = model(x)
    age = out[0, 0] if meta["no_gender"] else out[0, 2]
    return float(age) * (meta["max_age"] - meta["min_age"]) + meta["avg_age"]


if __name__ == "__main__":
    import sys
    import time
    from src.pipeline import imread_rgb
    t = time.time()
    for p in sys.argv[1:]:
        a = estimate(imread_rgb(p))
        assert a is None or 0 < a < 120, a
        print(f"{p}: {a if a is None else round(a, 1)}세 ({time.time() - t:.2f}초)", missing() or "")
        t = time.time()
