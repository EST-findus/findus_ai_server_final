"""공개된 가중치를 원래 출처에서 받는다 (새 PC 구축용). 이미 있는 파일은 건너뛴다.

    python scripts/download_assets.py            # 전부 (약 9GB)
    python scripts/download_assets.py --list     # 받을 목록만

받지 못하는 것 (공개 출처 없음, 직접 복사): weights/sam/best_encoder.pt, weights/kin/gender_dir.pt, data/hair_refs/
→ docs/[02]_구축_가이드.md 3장
"""
import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = "https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/pretrained_models/"
HAIRFAST = [
    "StyleGAN/ffhq.pt", "StyleGAN/ffhq_PCA.npz", "Rotate/rotate_best.pth", "Blending/checkpoint.pth",
    "PostProcess/pp_model.pth", "PostProcess/latent_avg.pt", "ArcFace/backbone_ir50.pth",
    "BiSeNet/face_parsing_79999_iter.pth", "BiSeNet/seg.pth",
    "FeatureStyleEncoder/143_enc.pth", "FeatureStyleEncoder/79999_iter.pth",
    "FeatureStyleEncoder/backbone.pth", "FeatureStyleEncoder/psp_ffhq_encode.pt",
    "encoder4editing/e4e_ffhq_encode.pt", "ShapeAdaptor/mask_generator.pth",
    "sean_checkpoints/CelebA-HQ_pretrained/latest_net_G.pth",
    "sean_checkpoints/CelebA-HQ_pretrained/latest_net_D.pth",
]
ITEMS = [  # (저장 경로, 종류, 출처)
    ("weights/sam/sam_ffhq_aging.pt", "gdrive", "1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC"),   # SAM 공식 (약 2.2GB)
    ("weights/ir_se50/model_ir_se50.pth", "gdrive", "1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn"),  # pSp 공식 IR-SE50
    ("weights/codeformer/codeformer_colorization.pth", "http",
     "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_colorization.pth"),
    ("weights/codeformer/codeformer.pth", "http",  # 복원 (POST /v1/restore, 골랐을 때만)
     "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"),
    ("weights/facexlib/detection_Resnet50_Final.pth", "http",
     "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"),
    # 나이 변환 신원 점수용 ArcFace (insightface buffalo_l 288MB zip 에서 이 파일만 꺼냄)
    ("weights/arcface/w600k_r50.onnx", "zip",
     "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip#w600k_r50.onnx"),
    *[(f"third_party/HairFastGAN/pretrained_models/{f}", "http", HF + f) for f in HAIRFAST],
]


def fetch(rel: str, kind: str, src: str):
    dst = ROOT / rel
    if dst.exists() and dst.stat().st_size > 0:
        print(f"있음   {rel}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    print(f"받는 중 {rel}", flush=True)
    if kind == "gdrive":
        import gdown
        if gdown.download(id=src, output=str(part), quiet=False) is None:
            raise RuntimeError(f"구글 드라이브에서 받지 못함 (하루 횟수 제한일 수 있음): https://drive.google.com/uc?id={src}")
    elif kind == "zip":
        import zipfile
        url, member = src.split("#")
        zpath = dst.with_suffix(".zip.part")
        urllib.request.urlretrieve(url, zpath)
        with zipfile.ZipFile(zpath) as z:
            name = next(n for n in z.namelist() if n.endswith(member))  # zip 안에 폴더가 있어도 찾는다
            part.write_bytes(z.read(name))
        zpath.unlink()
    else:
        urllib.request.urlretrieve(src, part)
    part.rename(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if not (ROOT / "third_party" / "HairFastGAN").exists():
        sys.exit("third_party/HairFastGAN 이 없습니다. 먼저 git clone 하세요 (docs/[02] 2장).")
    failed = []
    for rel, kind, src in ITEMS:
        if args.list:
            print(f"{rel}\n    ← {'Google Drive id ' + src if kind == 'gdrive' else src}")
            continue
        try:
            fetch(rel, kind, src)
        except Exception as e:
            failed.append(f"{rel}: {e}")
    for f in failed:
        print("  [실패]", f)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
