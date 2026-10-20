import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
THIRD_PARTY = ROOT / "third_party"
CONFIGS = ROOT / "configs"
EXPERIMENTS = ROOT / "experiments"


def add_third_party(name: str) -> Path:
    """third_party/<name> 을 sys.path 맨 앞에 추가한다 (SAM의 models/utils 등 패키지명 충돌 방지)."""
    path = THIRD_PARTY / name
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. README의 third_party 설치 단계를 확인하세요.")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def weight(*parts: str) -> Path:
    path = WEIGHTS.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f"가중치 없음: {path}\n  → python scripts/download_weights.py 로 받으세요.")
    return path
