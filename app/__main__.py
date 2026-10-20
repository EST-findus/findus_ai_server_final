"""python -m app  →  uvicorn 으로 서버 실행. GPU 가 하나라 worker 는 1개로 고정한다."""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")), workers=1)
