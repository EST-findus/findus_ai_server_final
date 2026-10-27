"""결과 저장소. 요청 하나 = 결과 하나 (outputs/<result_id>/). 한 번 만든 결과는 바꾸지 않는다.

    outputs/20261020-153012-a1b2c3/
        meta.json        {result_id, kind: restore|aging|hair|glasses|db, parent_id, sex, created, images: {키: 파일}, ...}
        age30.png        FFHQ 정렬 1024 결과 (무손실)

키는 나이 변환이면 'age30' 처럼 나이, 올린 사진에 바로 머리를 입히면 'face'. 복원은 'face'(복원)와 'input'(복원 전).
다음 단계(머리 → 안경)는 이전 결과의 result_id 를 받아 그 이미지들 위에 쌓는다.
DB 원본으로 만든 결과는 meta 에 origin(아동·원본 사진 id)을 남기고, 이어서 만든 결과는 origin 을 물려받는다.
"""
import json
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from .settings import OUTPUT_DIR, RESULT_TTL_HOURS

ID_RE = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}")
KEY_RE = re.compile(r"age\d{1,3}|face|input")


class NotFound(Exception):
    pass


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)


def _dir(result_id: str) -> Path:
    if not ID_RE.fullmatch(result_id or ""):  # 경로 조작 방지
        raise NotFound(f"잘못된 result_id: {result_id}")
    return OUTPUT_DIR / result_id


def save(kind: str, images: Dict[str, np.ndarray], sex: Optional[str], parent_id: Optional[str] = None,
         **extra) -> dict:
    if parent_id and "origin" not in extra:  # 머리·안경을 이어 붙여도 최초 DB 원본 연결을 잃지 않게
        origin = _origin_of(parent_id)
        if origin:
            extra["origin"] = origin
    rid = new_id()
    d = OUTPUT_DIR / rid
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    for key, img in images.items():
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not ok:
            raise IOError(f"PNG 저장 실패: {key}")
        buf.tofile(str(d / f"{key}.png"))
        files[key] = f"{key}.png"
    meta = {"result_id": rid, "kind": kind, "parent_id": parent_id, "sex": sex,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "images": files, **extra}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def meta(result_id: str) -> dict:
    f = _dir(result_id) / "meta.json"
    if not f.exists():
        raise NotFound(f"결과 없음: {result_id} (지워졌거나 잘못된 id)")
    return json.loads(f.read_text(encoding="utf-8"))


def _origin_of(result_id: str) -> Optional[dict]:
    return meta(result_id).get("origin")


def image_path(result_id: str, key: str) -> Path:
    if not KEY_RE.fullmatch(key or ""):
        raise NotFound(f"잘못된 이미지 키: {key}")
    m = meta(result_id)
    if key not in m["images"]:
        raise NotFound(f"{result_id} 에 '{key}' 이미지가 없습니다. 있는 것: {sorted(m['images'])}")
    return _dir(result_id) / m["images"][key]


def load_images(result_id: str, keys=None) -> Dict[str, np.ndarray]:
    m = meta(result_id)
    keys = keys or list(m["images"])
    out = {}
    for k in keys:
        p = image_path(result_id, k)
        out[k] = cv2.cvtColor(cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return out


_BASE_KEYS = {"result_id", "kind", "parent_id", "sex", "created", "images", "origin"}


def lineage(result_id: str) -> list:
    """처음 단계부터 이 결과까지 [{result_id, kind, options}]. 지워진 앞 단계는 빠진다."""
    chain, rid = [], result_id
    while rid:
        try:
            m = meta(rid)
        except NotFound:
            break
        chain.append({"result_id": rid, "kind": m["kind"],
                      "options": {k: v for k, v in m.items() if k not in _BASE_KEYS}})
        rid = m.get("parent_id")
    return chain[::-1]


def cleanup():
    """RESULT_TTL_HOURS 보다 오래된 결과 폴더를 지운다 (얼굴 사진을 오래 두지 않기)."""
    if RESULT_TTL_HOURS <= 0 or not OUTPUT_DIR.exists():
        return 0
    limit = time.time() - RESULT_TTL_HOURS * 3600
    n = 0
    for d in OUTPUT_DIR.iterdir():
        if d.is_dir() and ID_RE.fullmatch(d.name) and d.stat().st_mtime < limit:
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    return n
