"""서버 설정. 모두 환경 변수로 바꿀 수 있다 (.env 는 쓰지 않는다. 실행할 때 직접 넣는다)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "final.yaml"              # 나이 변환 조합 (ai_lab 최종 모델과 같은 파일)
HAIR_PRESETS = ROOT / "configs" / "hair_presets.yaml"  # 머리 스타일·색 목록
HAIR_REFS = ROOT / "data" / "hair_refs"                # <M|F>/<스타일 key>/*.png 참조 사진 (scripts/sync_assets.py)

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT / "outputs"))      # 결과 저장 (얼굴 사진이라 git 제외)
RESULT_TTL_HOURS = float(os.getenv("RESULT_TTL_HOURS", "24"))     # 이보다 오래된 결과는 지운다 (0 = 지우지 않음)
QUEUE_TIMEOUT = float(os.getenv("QUEUE_TIMEOUT", "300"))          # GPU 를 기다리는 최대 시간(초). 넘으면 503
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_AGES = int(os.getenv("MAX_AGES", "5"))                        # 한 번에 만들 수 있는 나이 개수
API_KEY = os.getenv("API_KEY") or None                            # 정하면 X-API-Key 헤더가 맞아야 한다
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
