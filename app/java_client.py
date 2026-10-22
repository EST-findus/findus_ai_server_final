"""Java 백엔드 연동 (작업 화면 /test/workbench 전용). DB 접속 정보는 갖지 않고 Java API 만 부른다.

    조회  GET  /api/missing-persons, /{id}, /{id}/images/{imageId}      기존 공개 API
    저장  POST /api/local/aiworkbench/saves                              로컬 전용 (Java 를 AI_WORKBENCH_ENABLED=true 로 실행)
          GET  /api/local/aiworkbench/saves, /ai-images/{id}

Java 주소는 서버 설정(JAVA_API_URL)으로만 정한다. 화면이 주소를 넘기지 않는다.
"""
import json
import secrets
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Optional, Tuple

from .settings import JAVA_API_URL, JAVA_TIMEOUT

# 시스템 프록시를 타지 않게 (로컬 Java 로 바로)
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class JavaError(Exception):
    """Java 가 거절했거나 연결이 안 됨. status 는 화면에 그대로 돌려줄 HTTP 상태."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status


def as_uuid(value: str, name: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except ValueError:
        raise JavaError(400, f"{name} 형식이 잘못됐습니다: {value}")


def _request(method: str, path: str, params: Optional[dict] = None, body: Optional[bytes] = None,
             content_type: Optional[str] = None) -> Tuple[int, str, bytes]:
    url = JAVA_API_URL + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, data=body, method=method, headers={"Content-Type": content_type} if content_type else {})
    try:
        with _opener.open(req, timeout=JAVA_TIMEOUT) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read())
            detail = data.get("message") or data.get("detail") or data.get("error") or e.reason
        except ValueError:
            detail = e.reason
        # 입력 문제(400·404·409)는 그대로, 나머지는 Java 쪽 문제로 본다
        raise JavaError(e.code if e.code in (400, 404, 409) else 502, f"Java {e.code}: {detail}")
    except (urllib.error.URLError, OSError) as e:
        raise JavaError(502, f"Java 백엔드에 연결하지 못했습니다 ({JAVA_API_URL}): {getattr(e, 'reason', e)}")


def _json(method: str, path: str, **kw):
    return json.loads(_request(method, path, **kw)[2])


# ---------------------------------------------------------------- 조회 (기존 API)

def persons(page: int, size: int) -> dict:
    return _json("GET", "/api/missing-persons", params={"page": page, "size": size})


def person(person_id: str) -> dict:
    return _json("GET", f"/api/missing-persons/{as_uuid(person_id, 'person_id')}")


def original_image(person_id: str, image_id: str) -> Tuple[bytes, str]:
    """원본 사진 바이트. 다른 아동의 사진·공개 중단이면 Java 가 404."""
    _, ctype, data = _request("GET", f"/api/missing-persons/{as_uuid(person_id, 'person_id')}/images/{as_uuid(image_id, 'source_image_id')}")
    return data, ctype or "image/jpeg"


# ---------------------------------------------------------------- 저장 (로컬 전용 API)

def save(fields: dict, png: bytes) -> Tuple[int, dict]:
    """→ (201 새로 저장 | 200 이미 저장됨, 저장 정보). 같은 result_id+key 로 다른 사진이면 409."""
    boundary = "----findus" + secrets.token_hex(12)
    parts = []
    for k, v in fields.items():
        if v is None:
            continue
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode("utf-8"))
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="result.png"\r\n'
                 f'Content-Type: image/png\r\n\r\n'.encode("utf-8") + png + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    status, _, data = _request("POST", "/api/local/aiworkbench/saves", body=b"".join(parts),
                               content_type=f"multipart/form-data; boundary={boundary}")
    return status, json.loads(data)


def saves(person_id: Optional[str] = None, result_id: Optional[str] = None) -> list:
    params = {"personId": as_uuid(person_id, "person_id")} if person_id else {"resultId": result_id}
    return _json("GET", "/api/local/aiworkbench/saves", params=params)


def ai_image(ai_image_id: str) -> Tuple[bytes, str]:
    _, ctype, data = _request("GET", f"/api/local/aiworkbench/ai-images/{as_uuid(ai_image_id, 'ai_image_id')}")
    return data, ctype or "image/png"
