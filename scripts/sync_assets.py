"""ai_lab 에서 서버에 필요한 가중치·외부 코드·머리 참조 사진만 복사한다. 여러 번 실행해도 안전하다 (같은 파일은 건너뜀).

    python scripts/sync_assets.py                       # 기본: ../ai_lab 에서 복사
    python scripts/sync_assets.py --lab D:/ai_lab       # ai_lab 위치가 다를 때
    python scripts/sync_assets.py --link                # 같은 디스크면 하드링크 (디스크를 더 쓰지 않음)
    python scripts/sync_assets.py --check               # 복사하지 않고 빠진 것만 확인

복사하는 것 (모두 .gitignore 대상. AI Hub 사진·가중치는 재배포 금지):
  weights/       configs/final.yaml 이 쓰는 가중치만 (SAM, 나이 인코더 best_encoder, 성별 방향, IR-SE50, RetinaFace, CodeFormer 컬러화)
  third_party/   SAM, CodeFormer(basicsr), HairFastGAN (patch_third_party.py 로 Windows 패치가 끝난 상태 그대로)
  data/hair_refs configs/hair_presets.yaml 의 styles 에 있는 스타일의 참조 사진 + 정렬·썸네일 캐시
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

WEIGHTS = [  # ai_lab 경로 또는 (ai_lab 경로, 서버 경로)
    "sam/sam_ffhq_aging.pt",                       # SAM 본체 (디코더·재구성 인코더)
    ("sam/sam_encoder_full_ft.pt", "sam/best_encoder.pt"),  # aging.encoder_full: 나이 인코더 전체 튜닝 (받은 best_encoder.pt 와 같은 가중치)
    "kin/gender_dir.pt",                           # aging.params.gender_dir: 성별 방향 + 판정기
    "ir_se50/model_ir_se50.pth",                   # 성별 그림 보고 고치기용 IR-SE50 (학습용 망)
    "facexlib/detection_Resnet50_Final.pth",       # RetinaFace 얼굴 검출
    "codeformer/codeformer_colorization.pth",      # 흑백 컬러화
]
THIRD_PARTY = {  # 이름 → 빼고 복사할 폴더·파일 (실행에 필요 없는 것)
    "SAM": {".git", "__pycache__", "notebooks", "docs", "environment"},
    "CodeFormer": {".git", "__pycache__", "assets", "inputs", "web-demos", "docs", "weights"},
    "HairFastGAN": {".git", "__pycache__", "docs"},
}


def same(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return False
    a, b = src.stat(), dst.stat()
    return a.st_size == b.st_size and int(a.st_mtime) <= int(b.st_mtime)


def put(src: Path, dst: Path, link: bool, check: bool, stats: dict):
    if not src.exists():
        stats["missing"].append(str(src))
        return
    if same(src, dst):
        stats["skip"] += 1
        return
    if check:
        stats["todo"].append(str(dst.relative_to(ROOT)))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if link:
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)
    stats["copy"] += 1
    stats["bytes"] += src.stat().st_size


def walk(src_dir: Path, skip: set):
    for p in src_dir.rglob("*"):
        rel = p.relative_to(src_dir)
        if p.is_file() and not (set(rel.parts) & skip):
            yield rel


def style_refs(presets: dict, ref_dir: Path, sex: str, key: str) -> list:
    """web/server.py _style_refs 와 같은 규칙: selected_refs 가 있으면 그 사진만, 없으면 폴더에서 exclude_refs 를 뺀다."""
    d = ref_dir / sex / key
    have = sorted(q.name for q in d.glob("*") if q.suffix.lower() in (".png", ".jpg", ".jpeg")) if d.exists() else []
    chosen = (presets.get("selected_refs") or {}).get(f"{sex}/{key}")
    if chosen:
        return [n for n in chosen if n in have]
    skip = set(presets.get("exclude_refs") or [])
    return [n for n in have if f"{sex}/{key}/{n}" not in skip]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", type=Path, default=ROOT.parent / "ai_lab", help="ai_lab 저장소 위치")
    ap.add_argument("--link", action="store_true", help="복사 대신 하드링크 (같은 디스크일 때만)")
    ap.add_argument("--check", action="store_true", help="복사하지 않고 빠진 것만 보여 준다")
    args = ap.parse_args()
    lab = args.lab.resolve()
    if not (lab / "src" / "pipeline.py").exists():
        sys.exit(f"ai_lab 저장소가 아닙니다: {lab}")
    stats = {"copy": 0, "skip": 0, "bytes": 0, "missing": [], "todo": []}
    put_ = lambda s, d: put(s, d, args.link, args.check, stats)  # noqa: E731

    for w in WEIGHTS:
        src, dst = w if isinstance(w, tuple) else (w, w)
        put_(lab / "weights" / src, ROOT / "weights" / dst)

    for name, skip in THIRD_PARTY.items():
        src_dir = lab / "third_party" / name
        if not src_dir.exists():
            stats["missing"].append(str(src_dir))
            continue
        for rel in walk(src_dir, skip):
            put_(src_dir / rel, ROOT / "third_party" / name / rel)

    presets = yaml.safe_load((ROOT / "configs" / "hair_presets.yaml").read_text(encoding="utf-8"))
    refs = lab / "data" / "hair_refs"
    for sex, styles in presets["styles"].items():
        for st in styles:
            for n in style_refs(presets, refs, sex, st["key"]):
                stem = Path(n).stem
                put_(refs / sex / st["key"] / n, ROOT / "data" / "hair_refs" / sex / st["key"] / n)
                for cache, ext in (("_aligned", ".png"), ("_hair_only", ".jpg")):  # 없으면 서버가 처음 쓸 때 만든다
                    c = refs / cache / sex / st["key"] / (stem + ext)
                    if c.exists():
                        put_(c, ROOT / "data" / "hair_refs" / cache / sex / st["key"] / (stem + ext))

    if args.check:
        print(f"준비됨 {stats['skip']}개, 복사할 것 {len(stats['todo'])}개")
        for t in stats["todo"][:30]:
            print("  +", t)
    else:
        print(f"복사 {stats['copy']}개 ({stats['bytes'] / 2**30:.2f} GB), 이미 있음 {stats['skip']}개")
    for m in stats["missing"]:
        print("  [없음]", m)
    sys.exit(1 if stats["missing"] else 0)


if __name__ == "__main__":
    main()
