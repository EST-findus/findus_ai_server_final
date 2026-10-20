"""FindUs 얼굴 변환 API 서버.

    python -m app                        # 시험 페이지 /test/aging (나이 변환), /test/post (머리·화장·안경), /docs (Swagger 문서)

    POST /v1/aging     사진 + 나이 → 그 나이 얼굴
    POST /v1/hair      나이 변환 결과(result_id) 또는 얼굴 사진 + 머리 스타일·색 → 머리 바꾼 얼굴
    POST /v1/glasses   (아직 구현 전, 501) 결과 또는 사진 + 안경 스타일 → 안경 씌운 얼굴
    GET  /v1/results/{result_id}                    결과 정보
    GET  /v1/results/{result_id}/images/{key}.png   결과 이미지
"""
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # HairFastGAN 이 잠깐 작업 폴더를 바꿔도 src 를 찾게 절대 경로로
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from src.hair_fit import USER_LIMIT  # noqa: E402

from . import store  # noqa: E402
from .engine import BadInput, Busy, NotReady, engine, presets, resolve_color, style_refs  # noqa: E402
from .schemas import GlassesStylesOut, HairStylesOut, HealthOut, ResultOut  # noqa: E402
from .settings import API_KEY, CORS_ORIGINS, MAX_AGES, MAX_UPLOAD_MB  # noqa: E402


@asynccontextmanager
async def lifespan(_):
    store.cleanup()
    yield


app = FastAPI(title="FindUs 얼굴 변환 API", version="0.1.0", lifespan=lifespan,
              description="ai_lab 최종 모델(configs/final.yaml)을 그대로 쓰는 나이 변환·머리 스타일 서버. "
                          "결과 이미지는 모두 FFHQ 정렬 1024x1024 PNG.")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


def _err(status: int):
    return lambda _req, e: JSONResponse(status_code=status, content={"detail": str(e)})


app.add_exception_handler(BadInput, _err(400))
app.add_exception_handler(store.NotFound, _err(404))
app.add_exception_handler(NotReady, _err(501))
app.add_exception_handler(Busy, _err(503))


def check_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "X-API-Key 가 맞지 않습니다.")


v1 = APIRouter(prefix="/v1", dependencies=[Depends(check_key)])


# ---------------------------------------------------------------- 도우미

def _decode(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    if len(data) > MAX_UPLOAD_MB * 2**20:
        raise BadInput(f"사진이 너무 큽니다 ({len(data) / 2**20:.1f}MB > {MAX_UPLOAD_MB:.0f}MB)")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise BadInput(f"이미지를 읽지 못했습니다: {upload.filename}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _sex(sex: Optional[str]) -> str:
    s = (sex or "").strip().upper()
    if s not in ("M", "F"):
        raise BadInput("sex 는 M 또는 F 입니다.")
    return s


def _source(result_id: Optional[str], keys: Optional[str], image: Optional[UploadFile], sex: Optional[str]):
    """다음 단계의 입력: 올린 사진(image) 또는 이전 결과(result_id). → (이미지들, 성별, 부모 id, 정렬 필요 여부)"""
    if image is not None and image.filename:
        return {"face": _decode(image)}, _sex(sex), None, True
    if result_id:
        m = store.meta(result_id)
        wanted = [k.strip() for k in keys.split(",") if k.strip()] if keys else None
        return store.load_images(result_id, wanted), _sex(sex or m.get("sex")), result_id, False
    raise BadInput("result_id(이전 결과) 또는 image(얼굴 사진) 중 하나가 필요합니다.")


def _result(meta: dict, seconds: float, info: Optional[dict] = None, extra: Optional[dict] = None) -> dict:
    rid = meta["result_id"]
    images = [{"key": k, "age": int(k[3:]) if k.startswith("age") else None,
               "url": f"/v1/results/{rid}/images/{k}.png", "info": (info or {}).get(k, {})}
              for k in meta["images"]]
    return {"result_id": rid, "kind": meta["kind"], "parent_id": meta.get("parent_id"), "sex": meta.get("sex"),
            "seconds": round(seconds, 1), "images": images, "extra": extra or {}}


# ---------------------------------------------------------------- 1. 나이 변환

@v1.post("/aging", response_model=ResultOut, summary="사진 + 나이 → 나이 변환 얼굴")
def aging(image: UploadFile = File(..., description="어릴 때 얼굴 사진 (jpg/png, 흑백이면 자동 컬러화)"),
          ages: str = Form(..., description="목표 나이. 하나(30) 또는 쉼표로 여러 개(20,30,40)", examples=["20,30,40"]),
          sex: str = Form(..., description="M | F. 결과의 성별이 뒤바뀌지 않게 보정한다"),
          gender_strength: float = Form(2.0, ge=0, le=5, description="성별 보정 세기 (0 = 끔, 기본 2)")):
    try:
        age_list = sorted({int(a) for a in ages.replace(" ", "").split(",") if a})
    except ValueError:
        raise BadInput(f"ages 는 숫자를 쉼표로 이은 것입니다: {ages}")
    if not age_list or len(age_list) > MAX_AGES or not all(1 <= a <= 100 for a in age_list):
        raise BadInput(f"ages 는 1~100 사이 숫자 1~{MAX_AGES}개입니다: {ages}")
    sex = _sex(sex)
    img = _decode(image)
    t0 = time.time()
    images, extra = engine.age(img, age_list, sex, gender_strength)
    extra.update(ages=age_list, gender_strength=gender_strength)
    meta = store.save("aging", images, sex, **extra)
    store.cleanup()
    return _result(meta, time.time() - t0, extra=extra)


# ---------------------------------------------------------------- 2. 머리 스타일

@v1.get("/hair/styles", response_model=HairStylesOut, summary="고를 수 있는 머리 스타일·색")
def hair_styles():
    pr = presets()
    return {"styles": {sex: [{**{k: st.get(k, "") for k in ("key", "name", "desc")}, "refs": style_refs(sex, st["key"])}
                             for st in styles] for sex, styles in pr["styles"].items()},
            "colors": [{k: c[k] for k in ("key", "name", "hex")} for c in pr["colors"]]}


@v1.get("/hair/styles/{sex}/{key}/{ref}/thumbnail", summary="참조 사진 썸네일 (얼굴은 지우고 머리만)")
def hair_thumbnail(sex: str, key: str, ref: str):
    if ref not in style_refs(sex, key):  # 목록에 있는 파일만 (임의 경로 접근 막기)
        raise store.NotFound(f"참조 사진 없음: {sex}/{key}/{ref}")
    return FileResponse(engine.thumbnail(sex, key, ref), media_type="image/jpeg")


@v1.post("/hair", response_model=ResultOut, summary="머리 스타일·색 적용")
def hair(result_id: Optional[str] = Form(None, description="이전 결과 id (나이 변환 결과 등). image 와 둘 중 하나"),
         keys: Optional[str] = Form(None, description="result_id 의 이미지 중 일부만 (예: age30,age40). 비우면 전부"),
         image: Optional[UploadFile] = File(None, description="얼굴 사진 (나이 변환 없이 바로 머리만 바꿀 때)"),
         sex: Optional[str] = Form(None, description="M | F. image 를 올렸을 때 필수 (result_id 면 그 결과의 성별)"),
         style: Optional[str] = Form(None, description="스타일 key (GET /v1/hair/styles). 비우면 색만 바꾼다"),
         ref: str = Form("auto", description="참조 사진 이름 또는 auto (얼굴 폭이 가장 비슷한 참조를 자동으로)"),
         color: Optional[str] = Form(None, description="색 key(예: ash_brown) 또는 #RRGGBB. 비우면 참조 사진 색"),
         scale: float = Form(1.0, description=f"머리 크기 배율 ({USER_LIMIT[0]}~{USER_LIMIT[1]})")):
    style = style or None
    color_hex = resolve_color(color)
    if not (style or color_hex):
        raise BadInput("style 이나 color 중 하나는 골라야 합니다.")
    if not USER_LIMIT[0] <= scale <= USER_LIMIT[1]:
        raise BadInput(f"scale 은 {USER_LIMIT[0]}~{USER_LIMIT[1]} 입니다: {scale}")
    images, sex, parent, align = _source(result_id, keys, image, sex)
    t0 = time.time()
    out, info = engine.hair(images, sex, style, ref, color_hex, scale, align)
    extra = {"style": style, "ref": ref if style else None, "color": color_hex, "scale": scale}
    meta = store.save("hair", out, sex, parent, **extra)
    return _result(meta, time.time() - t0, info, extra)


# ---------------------------------------------------------------- 3. 안경 (구현 전)

@v1.get("/glasses/styles", response_model=GlassesStylesOut, summary="안경 스타일 (구현 전이라 비어 있음)")
def glasses_styles():
    return {"implemented": False, "styles": []}


@v1.post("/glasses", response_model=ResultOut, summary="안경 씌우기 (구현 전: 501)",
         responses={501: {"description": "아직 구현 전"}})
def glasses(result_id: Optional[str] = Form(None, description="이전 결과 id (머리 결과 등). image 와 둘 중 하나"),
            keys: Optional[str] = Form(None, description="result_id 의 이미지 중 일부만. 비우면 전부"),
            image: Optional[UploadFile] = File(None, description="얼굴 사진"),
            sex: Optional[str] = Form(None, description="M | F. image 를 올렸을 때 필수"),
            style: str = Form(..., description="안경 스타일 key (GET /v1/glasses/styles)")):
    images, sex, parent, align = _source(result_id, keys, image, sex)
    t0 = time.time()
    out, info = engine.glasses(images, style, align)
    meta = store.save("glasses", out, sex, parent, style=style)
    return _result(meta, time.time() - t0, info, {"style": style})


# ---------------------------------------------------------------- 결과

@v1.get("/results/{result_id}", summary="결과 정보 (이미지 주소 포함)")
def result(result_id: str):
    m = store.meta(result_id)
    return {**m, "urls": {k: f"/v1/results/{result_id}/images/{k}.png" for k in m["images"]}}


@v1.get("/results/{result_id}/images/{key}.png", summary="결과 이미지 (PNG)")
def result_image(result_id: str, key: str):
    return FileResponse(store.image_path(result_id, key), media_type="image/png")


app.include_router(v1)


STATIC = Path(__file__).with_name("static")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/test", include_in_schema=False)
def test_page():
    return RedirectResponse("/test/aging")


@app.get("/test/aging", include_in_schema=False)
def test_aging():
    """시험 페이지 ①: 나이 변환. API 키는 페이지에서 넣는다."""
    return FileResponse(STATIC / "aging.html", headers={"Cache-Control": "no-cache"})


@app.get("/test/post", include_in_schema=False)
def test_post():
    """시험 페이지 ②: 후처리 (머리·화장·안경). ?result_id= 로 나이 변환 결과를 넘겨받는다."""
    return FileResponse(STATIC / "post.html", headers={"Cache-Control": "no-cache"})


@app.get("/health", response_model=HealthOut, summary="서버 상태")
def health():
    import torch
    cuda = torch.cuda.is_available()
    return {"status": "ok", "cuda": cuda, "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "busy": engine.busy, "loaded": engine.loaded}
