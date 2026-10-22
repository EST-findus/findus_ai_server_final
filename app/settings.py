"""서버 설정. 모두 환경 변수로 바꿀 수 있다. 저장소 폴더의 .env 에 적어 두어도 된다 (.env.example 참고).
이미 정한 환경 변수($env:, setx)가 .env 보다 우선한다."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_env(path: Path):
    """KEY=값 줄만 읽는다 (# 은 주석). 패키지 없이."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_env(ROOT / ".env")
CONFIG = ROOT / "configs" / "final.yaml"              # 나이 변환 조합 (ai_lab 최종 모델과 같은 파일)
HAIR_PRESETS = ROOT / "configs" / "hair_presets.yaml"  # 머리 스타일·색 목록
HAIR_REFS = ROOT / "data" / "hair_refs"                # <M|F>/<스타일 key>/*.png 참조 사진 (scripts/sync_assets.py)
ENCODER_DIR = ROOT / "weights" / "sam" / "encoders"    # 여기 넣은 나이 인코더(*.pt)를 화면·API 에서 고를 수 있다

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT / "outputs"))      # 결과 저장 (얼굴 사진이라 git 제외)
RESULT_TTL_HOURS = float(os.getenv("RESULT_TTL_HOURS", "24"))     # 이보다 오래된 결과는 지운다 (0 = 지우지 않음)
QUEUE_TIMEOUT = float(os.getenv("QUEUE_TIMEOUT", "300"))          # GPU 를 기다리는 최대 시간(초). 넘으면 503
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_AGES = int(os.getenv("MAX_AGES", "5"))                        # 한 번에 만들 수 있는 나이 개수
API_KEY = os.getenv("API_KEY") or None                            # 정하면 X-API-Key 헤더가 맞아야 한다
WORKBENCH_REMOTE = os.getenv("WORKBENCH_REMOTE", "").lower() == "true"  # true + API_KEY 면 DB 원본 작업을 다른 PC(Cloudflare 등)에서도 쓴다
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
JAVA_API_URL = os.getenv("JAVA_API_URL", "http://127.0.0.1:18080").rstrip("/")  # 실종아동 DB 를 가진 Java 백엔드 (작업 화면 전용)
JAVA_TIMEOUT = float(os.getenv("JAVA_TIMEOUT", "30"))                              # Java 요청 최대 대기(초)
