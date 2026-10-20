"""단계별 후보 모델 등록소. config의 이름만 바꾸면 모델을 갈아끼울 수 있다.

    @register("aging", "sam")
    class SAMAging: ...

    build("aging", "sam", device="cuda")
"""
STAGES = ("detector", "colorizer", "restorer", "upscaler", "aging", "evaluator")
REGISTRY = {stage: {} for stage in STAGES}


def register(stage: str, name: str):
    def deco(cls):
        REGISTRY[stage][name] = cls
        cls.name = name
        return cls
    return deco


def build(stage: str, name: str, **kwargs):
    _import_stages()
    if name not in REGISTRY[stage]:
        raise KeyError(f"[{stage}] '{name}' 없음. 사용 가능: {sorted(REGISTRY[stage])}")
    return REGISTRY[stage][name](**kwargs)


def available() -> dict:
    _import_stages()
    return {stage: sorted(names) for stage, names in REGISTRY.items()}


def _import_stages():
    from .stages import detect, colorize, restore, upscale, aging, evaluate  # noqa: F401  (등록 부작용)
