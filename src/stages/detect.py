"""1단계: 얼굴 검출 + 랜드마크.

모든 검출기는 RGB uint8 이미지를 받아 Face 리스트(큰 얼굴 순)를 돌려준다.
landmarks5 순서: [왼쪽 눈, 오른쪽 눈, 코끝, 왼쪽 입꼬리, 오른쪽 입꼬리] (이미지 기준 좌/우)
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..paths import WEIGHTS, weight
from ..registry import register


@dataclass
class Face:
    bbox: np.ndarray                         # [x1, y1, x2, y2]
    landmarks5: np.ndarray                   # (5, 2)
    score: float = 1.0
    landmarks68: Optional[np.ndarray] = None  # (68, 2), 68점 검출기만

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float((x2 - x1) * (y2 - y1))


def lm68_to_lm5(lm68: np.ndarray) -> np.ndarray:
    """iBUG 68점 → 5점. FFHQ 원본 정렬과 같은 점(눈 평균, 입꼬리 48/54)을 쓴다."""
    return np.stack([
        lm68[36:42].mean(0),
        lm68[42:48].mean(0),
        lm68[30],
        lm68[48],
        lm68[54],
    ]).astype(np.float32)


def _sort(faces: List[Face]) -> List[Face]:
    return sorted(faces, key=lambda f: f.area, reverse=True)


@register("detector", "retinaface")
class RetinaFaceDetector:
    """facexlib의 RetinaFace(ResNet50). 가중치는 첫 실행 때 weights/facexlib 로 자동 다운로드."""

    def __init__(self, device="cuda", threshold=0.9, **_):
        from facexlib.detection import init_detection_model
        self.net = init_detection_model("retinaface_resnet50", half=False, device=device,
                                        model_rootpath=str(WEIGHTS / "facexlib"))
        self.threshold = threshold

    def __call__(self, img_rgb: np.ndarray) -> List[Face]:
        import torch
        with torch.no_grad():
            dets = self.net.detect_faces(img_rgb[:, :, ::-1].copy(), self.threshold)  # BGR 입력
        faces = [Face(bbox=d[:4], score=float(d[4]), landmarks5=d[5:15].reshape(5, 2).astype(np.float32))
                 for d in dets]
        return _sort(faces)


@register("detector", "fan")
class FANDetector:
    """face-alignment(2D-FAN) 68점. 얼굴 박스는 내부 SFD 검출기가 찾는다."""

    def __init__(self, device="cuda", **_):
        import face_alignment
        lm_type = getattr(face_alignment.LandmarksType, "TWO_D", None) or face_alignment.LandmarksType._2D
        self.fa = face_alignment.FaceAlignment(lm_type, device=device, flip_input=False)

    def __call__(self, img_rgb: np.ndarray) -> List[Face]:
        results = self.fa.get_landmarks_from_image(img_rgb, return_bboxes=True)
        lms, _, boxes = results if results is not None else (None, None, None)
        if not lms:
            return []
        faces = [Face(bbox=np.asarray(b[:4]), score=float(b[4]) if len(b) > 4 else 1.0,
                      landmarks5=lm68_to_lm5(lm), landmarks68=lm.astype(np.float32))
                 for lm, b in zip(lms, boxes)]
        return _sort(faces)


@register("detector", "dlib")
class DlibDetector:
    """dlib HOG + 68점 shape predictor. FFHQ 원본 정렬 재현용 (설치: README 참고)."""

    def __init__(self, **_):
        import dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(weight("dlib", "shape_predictor_68_face_landmarks.dat")))

    def __call__(self, img_rgb: np.ndarray) -> List[Face]:
        faces = []
        for rect in self.detector(img_rgb, 1):
            shape = self.predictor(img_rgb, rect)
            lm = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
            faces.append(Face(bbox=np.array([rect.left(), rect.top(), rect.right(), rect.bottom()]),
                              landmarks5=lm68_to_lm5(lm), landmarks68=lm))
        return _sort(faces)


@register("detector", "scrfd")
class SCRFDDetector:
    """insightface SCRFD (buffalo_l/det_10g). insightface 패키지가 필요하다."""

    def __init__(self, device="cuda", **_):
        from insightface.app import FaceAnalysis
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name="buffalo_l", root=str(WEIGHTS / "insightface"),
                                allowed_modules=["detection"], providers=providers)
        self.app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))

    def __call__(self, img_rgb: np.ndarray) -> List[Face]:
        dets = self.app.get(img_rgb[:, :, ::-1].copy())
        return _sort([Face(bbox=d.bbox, score=float(d.det_score), landmarks5=d.kps.astype(np.float32)) for d in dets])
