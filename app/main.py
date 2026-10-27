"""FindUs 얼굴 변환 API 서버.

    python -m app                        # 시험 페이지 /test/aging (나이 변환), /test/post (머리·화장·안경), /docs (Swagger 문서)

    POST /v1/aging     사진 + 나이 → 그 나이 얼굴
    POST /v1/hair      나이 변환 결과(result_id) 또는 얼굴 사진 + 머리 스타일·색 → 머리 바꾼 얼굴
    POST /v1/glasses   (아직 구현 전, 501) 결과 또는 사진 + 안경 스타일 → 안경 씌운 얼굴
    GET  /v1/results/{result_id}                    결과 정보
    GET  /v1/results/{result_id}/images/{key}.png   결과 이미지
    GET  /v1/results/{result_id}/images/{key}/age   결과 이미지의 추정 나이 (MiVOLO, CPU, 버튼 누를 때만)

    /v1/workbench/*    DB 원본 작업 (시험 페이지 /test/workbench). Java 백엔드에서 원본을 받아 변환하고,
                       고른 결과만 Java 로 보내 DB 에 저장한다. 이 PC 에서 같은 출처 요청만 받는다.
"""
import json
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
from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from src.hair_fit import USER_LIMIT  # noqa: E402

from . import age_eval, java_client, store  # noqa: E402
from .engine import (BadInput, Busy, NotReady, default_encoder, encoders, engine, presets, resolve_color,  # noqa: E402
                     style_refs)
from .schemas import EncodersOut, GlassesStylesOut, HairStylesOut, HealthOut, ResultOut  # noqa: E402
from .settings import (API_KEY, CONFIG, CORS_ORIGINS, MAX_AGES, MAX_UPLOAD_MB, ROOT as APP_ROOT,  # noqa: E402
                       WORKBENCH_REMOTE)


@asynccontextmanager
async def lifespan(_):
    if WORKBENCH_REMOTE and not API_KEY:
        print("[경고] WORKBENCH_REMOTE=true 지만 API_KEY 가 없어 DB 원본 작업은 계속 이 PC 에서만 열립니다.")
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
app.add_exception_handler(java_client.JavaError, lambda _req, e: JSONResponse(status_code=e.status, content={"detail": str(e)}))


def check_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "X-API-Key 가 맞지 않습니다.")


v1 = APIRouter(prefix="/v1", dependencies=[Depends(check_key)])


# ---------------------------------------------------------------- 도우미

def _decode(upload: UploadFile) -> np.ndarray:
    return _decode_bytes(upload.file.read(), upload.filename)


def _decode_bytes(data: bytes, name: str) -> np.ndarray:
    if len(data) > MAX_UPLOAD_MB * 2**20:
        raise BadInput(f"사진이 너무 큽니다 ({len(data) / 2**20:.1f}MB > {MAX_UPLOAD_MB:.0f}MB)")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise BadInput(f"이미지를 읽지 못했습니다: {name}")
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
            "seconds": round(seconds, 1), "images": images, "extra": extra or {}, "origin": meta.get("origin")}


def _ages(ages: str) -> list:
    try:
        age_list = sorted({int(a) for a in ages.replace(" ", "").split(",") if a})
    except ValueError:
        raise BadInput(f"ages 는 숫자를 쉼표로 이은 것입니다: {ages}")
    if not age_list or len(age_list) > MAX_AGES or not all(1 <= a <= 100 for a in age_list):
        raise BadInput(f"ages 는 1~100 사이 숫자 1~{MAX_AGES}개입니다: {ages}")
    return age_list


def _run_aging(img: np.ndarray, age_list: list, sex: str, strength: float, gender_strength: float,
               encoder: Optional[str] = None, original: Optional[np.ndarray] = None, parent_id: Optional[str] = None,
               **meta_extra) -> dict:
    t0 = time.time()
    images, extra = engine.age(img, age_list, sex, gender_strength, strength=strength, encoder=encoder, original=original)
    extra.update(ages=age_list, strength=strength, gender_strength=gender_strength, restore=original is not None)
    meta = store.save("aging", images, sex, parent_id, **extra, **meta_extra)
    store.cleanup()
    return _result(meta, time.time() - t0, extra=extra)


def _restored(result_id: str) -> tuple:
    """복원 결과 → (복원 얼굴, 복원 전 얼굴, meta)."""
    m = store.meta(result_id)
    if m["kind"] != "restore":
        raise BadInput(f"복원 결과가 아닙니다: {result_id} ({m['kind']})")
    imgs = store.load_images(result_id, ["face", "input"])
    return imgs["face"], imgs["input"], m


def _run_restore(img: np.ndarray, w: float, **meta_extra) -> dict:
    t0 = time.time()
    images, extra = engine.restore(img, w)
    meta = store.save("restore", images, None, **extra, **meta_extra)
    store.cleanup()
    return _result(meta, time.time() - t0, extra=extra)


# ---------------------------------------------------------------- 0. 복원 (골랐을 때만)

@v1.post("/restore", response_model=ResultOut, summary="사진 → CodeFormer 복원 얼굴 (보고 고른 뒤 나이 변환에 넣는다)")
def restore(image: UploadFile = File(..., description="어릴 때 얼굴 사진 (jpg/png, 흑백이면 자동 컬러화 후 복원)"),
            w: float = Form(0.5, ge=0, le=1, allow_inf_nan=False, description="CodeFormer 충실도 (0 = 강하게 보정, 1 = 원본 유지, 기본 0.5)")):
    return _run_restore(_decode(image), w)


# ---------------------------------------------------------------- 1. 나이 변환

@v1.post("/aging", response_model=ResultOut, summary="사진 + 나이 → 나이 변환 얼굴")
def aging(image: Optional[UploadFile] = File(None, description="어릴 때 얼굴 사진 (jpg/png, 흑백이면 자동 컬러화). result_id 와 둘 중 하나"),
          result_id: Optional[str] = Form(None, description="복원 결과 id (POST /v1/restore). 주면 복원 얼굴로 변환하고 원본 대비 점수도 낸다"),
          ages: str = Form(..., description="목표 나이. 하나(30) 또는 쉼표로 여러 개(20,30,40)", examples=["20,30,40"]),
          sex: str = Form(..., description="M | F. 결과의 성별이 뒤바뀌지 않게 보정한다"),
          strength: float = Form(1.0, ge=0, le=2, allow_inf_nan=False,
                                 description="나이 변환 강도 (0~2, 기본 1). 0은 나이 변화만 끄며 얼굴 재구성·성별 보정은 적용된다"),
          gender_strength: float = Form(2.0, ge=0, le=5, description="성별 보정 세기 (0 = 끔, 기본 2)"),
          encoder: Optional[str] = Form(None, description="나이 인코더 파일 이름 (GET /v1/aging/encoders). 비우면 기본")):
    age_list = _ages(ages)
    sex = _sex(sex)
    if result_id:
        face, original, _ = _restored(result_id)
        return _run_aging(face, age_list, sex, strength, gender_strength, encoder=encoder or None,
                          original=original, parent_id=result_id)
    if image is None or not image.filename:
        raise BadInput("image(얼굴 사진) 또는 result_id(복원 결과) 중 하나가 필요합니다.")
    return _run_aging(_decode(image), age_list, sex, strength, gender_strength, encoder=encoder or None)


@v1.get("/aging/encoders", response_model=EncodersOut, summary="고를 수 있는 나이 인코더 (weights/sam/)")
def aging_encoders():
    return {"default": default_encoder(), "encoders": encoders()}


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


@v1.get("/results/{result_id}/images/{key}/age", summary="결과 이미지의 추정 나이 (MiVOLO, CPU, 평가용)")
def result_age(result_id: str, key: str):
    img = store.load_images(result_id, [key])[key]
    t0 = time.time()
    age = age_eval.estimate(img)
    if age is None:
        raise BadInput(f"추정 나이를 쓸 수 없습니다: {age_eval.missing()}")
    return {"age": round(age, 1), "seconds": round(time.time() - t0, 2)}


# ---------------------------------------------------------------- 4. DB 원본 작업 (Java 연동)

def local_only(request: Request):
    """이 PC 에서, 이 서버가 띄운 화면에서 온 요청만. 다른 웹사이트가 몰래 저장 요청을 보내지 못하게 한다.
    WORKBENCH_REMOTE=true 이고 API_KEY 를 정했으면 다른 PC 도 받는다 (키는 앞의 check_key 가 이미 확인함)."""
    remote = (request.client.host if request.client else "") not in ("127.0.0.1", "::1", "localhost")
    if remote and not (WORKBENCH_REMOTE and API_KEY):
        raise HTTPException(403, "이 PC 에서만 쓸 수 있습니다. 다른 PC 에서 쓰려면 서버에 "
                                 "WORKBENCH_REMOTE=true 와 API_KEY 를 함께 정하세요 (.env.example).")
    origin = request.headers.get("origin")
    if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
        raise HTTPException(403, "다른 사이트에서 온 요청은 받지 않습니다.")


wb = APIRouter(prefix="/v1/workbench", tags=["DB 원본 작업"], dependencies=[Depends(check_key), Depends(local_only)])
MODEL = CONFIG.relative_to(APP_ROOT).as_posix()


@wb.get("/persons", summary="실종아동 목록 (Java 기존 API, 페이지 방식)")
def wb_persons(page: int = 0, size: int = 12):
    return java_client.persons(page, size)


@wb.get("/persons/{person_id}", summary="실종아동 상세와 원본 사진 목록")
def wb_person(person_id: str):
    return java_client.person(person_id)


@wb.get("/persons/{person_id}/images/{image_id}", summary="원본 사진 (화면 표시용, 저장하지 않음)")
def wb_original(person_id: str, image_id: str):
    data, ctype = java_client.original_image(person_id, image_id)
    return Response(data, media_type=ctype, headers={"Cache-Control": "no-store"})


def _wb_original(person_id: str, source_image_id: str) -> tuple:
    """DB 원본 사진과 결과에 남길 연결 정보."""
    p = java_client.person(person_id)
    data, _ = java_client.original_image(person_id, source_image_id)  # 다른 아동의 사진이면 Java 가 404
    origin = {"person_id": p["id"], "source_image_id": java_client.as_uuid(source_image_id, "source_image_id"),
              "person_name": p.get("name"), "model": MODEL}
    return _decode_bytes(data, "DB 원본"), origin


@wb.post("/restore", response_model=ResultOut, summary="DB 원본 → CodeFormer 복원 얼굴 (보고 고른 뒤 나이 변환에 넣는다)")
def wb_restore(person_id: str = Form(..., description="실종아동 id (Java DB UUID)"),
               source_image_id: str = Form(..., description="그 아동의 원본 사진 id"),
               w: float = Form(0.5, ge=0, le=1, allow_inf_nan=False, description="CodeFormer 충실도 (0 = 강하게 보정, 1 = 원본 유지, 기본 0.5)")):
    img, origin = _wb_original(person_id, source_image_id)
    return _run_restore(img, w, origin=origin)


@wb.post("/aging", response_model=ResultOut, summary="DB 원본 + 나이 → 나이 변환 (결과에 원본 연결 정보를 남긴다)")
def wb_aging(person_id: str = Form(..., description="실종아동 id (Java DB UUID)"),
             source_image_id: str = Form(..., description="그 아동의 원본 사진 id"),
             ages: str = Form(..., examples=["20,30,40"]),
             sex: str = Form(..., description="M | F"),
             strength: float = Form(1.0, ge=0, le=2, allow_inf_nan=False),
             gender_strength: float = Form(2.0, ge=0, le=5),
             encoder: Optional[str] = Form(None, description="나이 인코더 파일 이름 (GET /v1/aging/encoders). 비우면 기본"),
             result_id: Optional[str] = Form(None, description="같은 원본으로 만든 복원 결과 id (POST /v1/workbench/restore). 주면 복원 얼굴로 변환")):
    age_list = _ages(ages)
    sex = _sex(sex)
    encoder = encoder or None
    if result_id:
        face, original, m = _restored(result_id)
        o = m.get("origin")
        same = o and all(java_client.as_uuid(a, n) == java_client.as_uuid(b, n) for a, b, n in
                         ((o["person_id"], person_id, "person_id"), (o["source_image_id"], source_image_id, "source_image_id")))
        if not same:
            raise BadInput(f"복원 결과 {result_id} 는 이 아동·원본 사진으로 만든 것이 아닙니다.")
        origin = {**o, "encoder": encoder or default_encoder(), "restore": True}
        return _run_aging(face, age_list, sex, strength, gender_strength, encoder=encoder,
                          original=original, parent_id=result_id, origin=origin)
    img, origin = _wb_original(person_id, source_image_id)
    origin.update(encoder=encoder or default_encoder(), restore=False)
    return _run_aging(img, age_list, sex, strength, gender_strength, encoder=encoder, origin=origin)


@wb.post("/saves", summary="선택한 결과 저장 (Java 를 거쳐 DB 에 보관)",
         responses={201: {"description": "새로 저장"}, 409: {"description": "같은 결과가 다른 사진으로 이미 저장됨"}})
def wb_save(response: Response, result_id: str = Form(...), key: str = Form(..., description="이미지 키 (예: age30)")):
    m = store.meta(result_id)
    origin = m.get("origin")
    if not origin:
        raise BadInput("DB 원본으로 만든 결과만 저장할 수 있습니다.")
    if not key.startswith("age"):
        raise BadInput(f"목표 나이를 알 수 없는 이미지입니다: {key}")
    png = store.image_path(result_id, key).read_bytes()
    options = {"model": origin.get("model"), "encoder": origin.get("encoder"), "restore": origin.get("restore", False),
               "steps": store.lineage(result_id)}
    status, saved = java_client.save({"personId": origin["person_id"], "sourceImageId": origin["source_image_id"],
                                      "targetAge": int(key[3:]), "resultId": result_id, "imageKey": key,
                                      "options": json.dumps(options, ensure_ascii=False)}, png)
    response.status_code = status
    return saved


@wb.get("/saves", summary="저장 이력 (person_id 또는 result_id)")
def wb_saves(person_id: Optional[str] = None, result_id: Optional[str] = None):
    if not (person_id or result_id):
        raise BadInput("person_id 또는 result_id 가 필요합니다.")
    return java_client.saves(person_id, result_id)


@wb.get("/ai-images/{ai_image_id}", summary="DB 에 저장한 AI 사진")
def wb_ai_image(ai_image_id: str):
    data, ctype = java_client.ai_image(ai_image_id)
    return Response(data, media_type=ctype, headers={"Cache-Control": "no-store"})


@wb.post("/ai-images/{ai_image_id}/import", response_model=ResultOut,
         summary="DB 에 저장한 AI 사진 → 후처리 입력 (result_id 를 /v1/hair·/v1/glasses 에 넘긴다)")
def wb_import(ai_image_id: str, person_id: str = Form(..., description="실종아동 id (Java DB UUID)"),
              sex: str = Form(..., description="M | F")):
    aid = java_client.as_uuid(ai_image_id, "ai_image_id")
    p = java_client.person(person_id)
    s = next((s for s in java_client.saves(person_id) if s["aiImageId"] == aid), None)  # 다른 아동의 사진은 막는다
    if not s:
        raise store.NotFound(f"이 아동이 저장한 AI 사진이 아닙니다: {ai_image_id}")
    data, _ = java_client.ai_image(aid)
    try:
        opts = json.loads(s.get("options") or "{}")
    except ValueError:
        opts = {}
    origin = {"person_id": p["id"], "source_image_id": s["sourceImageId"], "person_name": p.get("name"),
              "model": opts.get("model") or MODEL, "encoder": opts.get("encoder"), "restore": opts.get("restore", False)}
    t0 = time.time()
    extra = {"ai_image_id": aid, "from_result": s.get("resultId")}
    meta = store.save("db", {f"age{s['targetAge']}": _decode_bytes(data, "DB AI 사진")}, _sex(sex), origin=origin, **extra)
    return _result(meta, time.time() - t0, extra=extra)


app.include_router(v1)
app.include_router(wb)


STATIC = Path(__file__).with_name("static")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
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


@app.get("/test/workbench", include_in_schema=False)
def test_workbench():
    """시험 페이지 ③: DB 원본 작업. 실종아동 원본으로 나이 변환 → 고른 결과만 DB 저장."""
    return FileResponse(STATIC / "workbench.html", headers={"Cache-Control": "no-cache"})


@app.get("/health", response_model=HealthOut, summary="서버 상태")
def health():
    import torch
    cuda = torch.cuda.is_available()
    return {"status": "ok", "cuda": cuda, "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "busy": engine.busy, "loaded": engine.loaded}
