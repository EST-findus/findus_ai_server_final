"""DB 원본 작업(/v1/workbench) 점검. GPU·Java 없이 돈다: 가짜 Java 서버를 띄우고 나이 변환은 가짜로 바꾼다.

    .venv\\Scripts\\python scripts\\check_workbench.py

확인하는 것: 원본 조회 중계, 생성만 하면 저장 안 됨, origin 기록·상속, 선택한 결과만 저장(재요청 200),
업로드 결과는 저장 거절, 다른 사이트 요청 403, Java 가 꺼졌을 때 502.
"""
import email
import email.policy
import json
import os
import re
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PERSON, SOURCE, OTHER = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
_, PNG = cv2.imencode(".png", np.full((8, 8, 3), 120, np.uint8))
PNG = PNG.tobytes()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeJava(BaseHTTPRequestHandler):
    """Java 백엔드의 조회·저장 API 를 흉내 낸다. 받은 저장 요청은 saves 에 쌓인다."""
    saves = {}
    requests = []

    def log_message(self, *a):
        pass

    def _send(self, status, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        FakeJava.requests.append(("GET", self.path, dict(self.headers)))
        person = {"id": PERSON, "name": "가상아동", "sex": "여자", "age": 5, "ageNow": 25,
                  "images": [{"id": SOURCE, "sortOrder": 0, "url": f"/api/missing-persons/{PERSON}/images/{SOURCE}"}]}
        if self.path.startswith("/api/missing-persons?"):
            return self._send(200, {"content": [person], "page": 0, "size": 12, "totalElements": 1, "totalPages": 1})
        if self.path == f"/api/missing-persons/{PERSON}":
            return self._send(200, person)
        if self.path == f"/api/missing-persons/{PERSON}/images/{SOURCE}":
            return self._send(200, PNG, "image/png")
        if self.path.startswith("/api/local/aiworkbench/saves?"):
            return self._send(200, list(FakeJava.saves.values()))
        m = re.fullmatch(r"/api/local/aiworkbench/ai-images/(.+)", self.path)
        if m and any(s["aiImageId"] == m.group(1) for s in FakeJava.saves.values()):
            return self._send(200, PNG, "image/png")
        self._send(404, {"code": "NOT_FOUND", "message": "사진을 찾을 수 없습니다."})

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        FakeJava.requests.append(("POST", self.path, dict(self.headers)))
        msg = email.message_from_bytes(b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body,
                                       policy=email.policy.default)
        fields = {p.get_param("name", header="content-disposition"): p.get_content() for p in msg.iter_parts()}
        key = (fields["resultId"], fields["imageKey"])
        if key in FakeJava.saves:
            return self._send(200, {**FakeJava.saves[key], "created": False})
        FakeJava.saves[key] = {"aiImageId": str(uuid.uuid4()), "personId": fields["personId"],
                               "sourceImageId": fields["sourceImageId"], "targetAge": int(fields["targetAge"]),
                               "resultId": key[0], "imageKey": key[1], "options": fields["options"],
                               "imageBytes": len(fields["image"]), "created": True}
        self._send(201, FakeJava.saves[key])


JAVA_PORT, APP_PORT = free_port(), free_port()
os.environ["JAVA_API_URL"] = f"http://127.0.0.1:{JAVA_PORT}"
os.environ["OUTPUT_DIR"] = tempfile.mkdtemp(prefix="workbench-check-")
os.environ.pop("API_KEY", None)

from app import main, store  # noqa: E402  (환경 변수를 먼저 정해야 설정에 반영된다)


def fake_age(img, ages, sex, gender_strength, strength=1.0, encoder=None):
    return {f"age{a}": np.full((16, 16, 3), a, np.uint8) for a in ages}, {"colorized": False}


main.engine.age = fake_age


def call(method, path, form=None, headers=None):
    data = None
    hdrs = dict(headers or {})
    if form is not None:
        data = urllib.parse.urlencode(form).encode()
        hdrs["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(f"http://127.0.0.1:{APP_PORT}{path}", data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read()
            return r.status, (json.loads(body) if r.headers.get_content_type() == "application/json" else body)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


class WorkbenchCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.java = ThreadingHTTPServer(("127.0.0.1", JAVA_PORT), FakeJava)
        threading.Thread(target=cls.java.serve_forever, daemon=True).start()
        import uvicorn
        cls.server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=APP_PORT, log_level="warning"))
        threading.Thread(target=cls.server.run, daemon=True).start()
        for _ in range(100):
            if cls.server.started:
                break
            time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.java.shutdown()

    def setUp(self):
        FakeJava.saves.clear()
        FakeJava.requests.clear()

    def aging(self, ages="30,40"):
        status, res = call("POST", "/v1/workbench/aging", {"person_id": PERSON, "source_image_id": SOURCE,
                                                            "ages": ages, "sex": "F"})
        self.assertEqual(status, 200, res)
        return res

    def test_원본_조회를_Java_로_중계한다(self):
        status, page = call("GET", "/v1/workbench/persons?page=0&size=12")
        self.assertEqual((status, page["content"][0]["name"]), (200, "가상아동"))
        status, data = call("GET", f"/v1/workbench/persons/{PERSON}/images/{SOURCE}")
        self.assertEqual((status, data), (200, PNG))
        status, _ = call("GET", f"/v1/workbench/persons/{PERSON}/images/{OTHER}")
        self.assertEqual(status, 404)
        status, _ = call("GET", "/v1/workbench/persons/../../etc")
        self.assertIn(status, (400, 404))

    def test_생성만_하면_저장하지_않고_origin_을_남긴다(self):
        res = self.aging()
        self.assertEqual(res["origin"]["person_id"], PERSON)
        self.assertEqual(res["origin"]["source_image_id"], SOURCE)
        self.assertEqual(res["origin"]["person_name"], "가상아동")
        self.assertEqual(res["origin"]["model"], "configs/final.yaml")
        self.assertFalse(any(m == "POST" for m, _, _ in FakeJava.requests))
        self.assertEqual(FakeJava.saves, {})

    def test_후처리_결과도_최초_원본_연결을_물려받는다(self):
        res = self.aging()
        hair = store.save("hair", store.load_images(res["result_id"], ["age30"]), "F", res["result_id"], style="bob")
        self.assertEqual(hair["origin"], res["origin"])
        steps = store.lineage(hair["result_id"])
        self.assertEqual([s["kind"] for s in steps], ["aging", "hair"])
        self.assertEqual(steps[1]["options"]["style"], "bob")

    def test_선택한_결과만_저장하고_재요청은_기존_결과를_돌려준다(self):
        res = self.aging()
        status, saved = call("POST", "/v1/workbench/saves", {"result_id": res["result_id"], "key": "age30"})
        self.assertEqual(status, 201, saved)
        self.assertEqual(list(FakeJava.saves), [(res["result_id"], "age30")])  # age40 은 저장 안 됨
        sent = FakeJava.saves[(res["result_id"], "age30")]
        self.assertEqual((sent["personId"], sent["sourceImageId"], sent["targetAge"]), (PERSON, SOURCE, 30))
        self.assertEqual(json.loads(sent["options"])["steps"][0]["options"]["ages"], [30, 40])
        status, again = call("POST", "/v1/workbench/saves", {"result_id": res["result_id"], "key": "age30"})
        self.assertEqual((status, again["created"], again["aiImageId"]), (200, False, saved["aiImageId"]))
        status, data = call("GET", f"/v1/workbench/ai-images/{saved['aiImageId']}")
        self.assertEqual(status, 200)

    def test_업로드로_만든_결과는_저장하지_않는다(self):
        meta = store.save("aging", {"age30": np.zeros((16, 16, 3), np.uint8)}, "M")
        status, res = call("POST", "/v1/workbench/saves", {"result_id": meta["result_id"], "key": "age30"})
        self.assertEqual(status, 400, res)
        self.assertEqual(FakeJava.saves, {})

    def test_다른_사이트에서_온_요청은_막는다(self):
        res = self.aging()
        form = {"result_id": res["result_id"], "key": "age30"}
        status, _ = call("POST", "/v1/workbench/saves", form, {"Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(FakeJava.saves, {})
        status, _ = call("POST", "/v1/workbench/saves", form, {"Origin": f"http://127.0.0.1:{APP_PORT}"})
        self.assertEqual(status, 201)
        self.assertNotIn("Origin", FakeJava.requests[-1][2])  # Java 로는 브라우저 헤더를 넘기지 않는다

    def test_Java_가_꺼져_있으면_502(self):
        old = main.java_client.JAVA_API_URL
        main.java_client.JAVA_API_URL = f"http://127.0.0.1:{free_port()}"
        try:
            status, res = call("GET", "/v1/workbench/persons")
        finally:
            main.java_client.JAVA_API_URL = old
        self.assertEqual(status, 502)
        self.assertIn("연결하지 못했습니다", res["detail"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
