"""응답 형식 (Swagger 문서 /docs 에 그대로 나온다)."""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ImageOut(BaseModel):
    key: str = Field(description="이미지 키. 나이 변환이면 'age30', 올린 사진에 바로 적용했으면 'face'")
    age: Optional[int] = Field(None, description="목표 나이 (나이 변환 결과에서 이어졌을 때)")
    url: str = Field(description="PNG 주소 (FFHQ 정렬 1024x1024)")
    info: Dict = Field(default_factory=dict, description="단계별 정보 (머리: 쓴 참조 ref, 배율 scale, 색 차이 color_dE. "
                                                         "안경: method, changed_ratio(바뀐 픽셀 비율). gan: bias(안경 세기), frame_area(테 면적), lens_changed(렌즈 안쪽 바뀐 비율). overlay: had_glasses, eyes, width_ratio)")


class ResultOut(BaseModel):
    result_id: str = Field(description="이 결과의 id. 다음 단계(머리·안경) 요청에 넣으면 그 위에 쌓인다")
    kind: str = Field(description="restore | aging | hair | glasses | db (DB 에서 가져온 AI 사진)")
    parent_id: Optional[str] = Field(None, description="이어서 만든 이전 결과의 id")
    sex: Optional[str] = None
    seconds: float = Field(description="처리 시간 (GPU 대기 포함)")
    images: List[ImageOut]
    extra: Dict = Field(default_factory=dict, description="나이 변환: colorized(흑백이라 컬러화했는가), identity(입력 대비 ArcFace), restore(복원 얼굴로 했는가), identity_original(복원 전 원본 대비), 옵션 값 등. 복원: identity.face(원본 대비)")
    origin: Optional[Dict] = Field(None, description="DB 원본으로 만든 결과의 연결 정보 (person_id, source_image_id, person_name, model, encoder). "
                                                     "이것이 있는 결과만 DB 저장 대상")


class EncodersOut(BaseModel):
    default: str = Field(description="encoder 를 비웠을 때 쓰는 나이 인코더 (configs/final.yaml)")
    encoders: List[str] = Field(description="weights/sam/ 에 넣은 파일 이름 (SAM 본체·기본 제외). POST /v1/aging 의 encoder 로 고른다")


class HairStyle(BaseModel):
    key: str
    name: str
    desc: str = ""
    refs: List[str] = Field(description="참조 사진 이름. ref 로 하나를 고르거나 'auto' (얼굴 폭이 가장 비슷한 것 자동)")


class HairColor(BaseModel):
    key: str
    name: str
    hex: str


class HairStylesOut(BaseModel):
    styles: Dict[str, List[HairStyle]] = Field(description="성별(M|F) → 스타일 목록")
    colors: List[HairColor]


class GlassesStyle(BaseModel):
    key: str = Field(description="POST /v1/glasses 의 style")
    name: str
    required: bool = Field(False, description="꼭 있어야 하는 안경 (검정 뿔테·검정 동글이)")
    preview: str = Field(description="미리보기 PNG 주소 (얼굴 없음)")


class GlassesMethod(BaseModel):
    name: str
    available: bool = Field(description="가중치·코드가 있어 쓸 수 있는가")
    styles: List[GlassesStyle]
    scale: Optional[List[float]] = Field(None, description="overlay: scale 범위")
    fine: Optional[List[float]] = Field(None, description="gan: size·thickness 범위")


class GlassesStylesOut(BaseModel):
    implemented: bool
    default_method: str = Field(description="method 를 비웠을 때 쓰는 방식")
    methods: Dict[str, GlassesMethod] = Field(description="gan(GlassesGAN 생성) | overlay(사진 합성)")


class HealthOut(BaseModel):
    status: str
    cuda: bool
    gpu: Optional[str] = None
    busy: bool = Field(description="지금 GPU 작업 중인가")
    loaded: List[str] = Field(description="GPU 에 올라가 있는 모델 (hair, aging, glasses)")
