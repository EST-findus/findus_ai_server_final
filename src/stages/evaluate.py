"""5단계: 같은 사람인지 평가. ArcFace 정렬(112x112) 얼굴 → L2 정규화 임베딩.

similarity(a, b) = 코사인 유사도 (-1~1, 높을수록 같은 사람).
"""
import importlib.util

import numpy as np

from ..paths import THIRD_PARTY, weight
from ..registry import register
from ._stub import stub


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


@register("evaluator", "arcface")
class ArcFaceEvaluator:
    """insightface buffalo_l 의 w600k_r50.onnx (ResNet50, WebFace600K)."""

    def __init__(self, **_):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(weight("arcface", "w600k_r50.onnx")),
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def embed(self, face112_rgb: np.ndarray) -> np.ndarray:
        x = (face112_rgb.astype(np.float32) - 127.5) / 127.5
        x = x.transpose(2, 0, 1)[None]
        return self.sess.run(None, {self.input_name: x})[0][0]


@register("evaluator", "adaface")
class AdaFaceEvaluator:
    """mk-minchul/AdaFace IR-101 (WebFace12M). 입력은 BGR 정규화."""

    def __init__(self, device="cuda", arch="ir_101", **_):
        import torch
        self.torch = torch
        self.device = device
        # AdaFace 저장소의 face_alignment 폴더가 pip face_alignment 와 이름이 겹치므로 net.py 만 직접 로드
        spec = importlib.util.spec_from_file_location("adaface_net", THIRD_PARTY / "AdaFace" / "net.py")
        net = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(net)
        self.model = net.build_model(arch)
        state = torch.load(weight("adaface", "adaface_ir101_webface12m.ckpt"), map_location="cpu")["state_dict"]
        self.model.load_state_dict({k[6:]: v for k, v in state.items() if k.startswith("model.")})
        self.model.to(device).eval()

    def embed(self, face112_rgb: np.ndarray) -> np.ndarray:
        torch = self.torch
        bgr = (face112_rgb[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5
        x = torch.from_numpy(bgr.transpose(2, 0, 1).copy())[None].to(self.device)
        with torch.no_grad():
            feat, _ = self.model(x)
        return feat[0].cpu().numpy()


stub("evaluator", "mtlface", "https://github.com/Hzzone/MTLFace")
stub("evaluator", "magface", "https://github.com/IrvingMeng/MagFace")
