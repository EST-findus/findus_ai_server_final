from ..registry import register


def stub(stage: str, name: str, repo: str):
    """후보로 등록만 해두고 아직 연결하지 않은 모델. config에 넣으면 안내 메시지와 함께 실패한다."""

    @register(stage, name)
    class _Stub:
        connected = False

        def __init__(self, **_):
            raise NotImplementedError(
                f"[{stage}] '{name}' 는 아직 연결 전입니다. {repo} 를 third_party/ 에 받고 "
                f"src/stages/{stage}.py 에 래퍼를 구현하세요."
            )

    _Stub.__name__ = f"Stub_{name}"
    return _Stub
