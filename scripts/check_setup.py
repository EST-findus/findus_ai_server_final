"""서버 실행에 필요한 외부 코드·가중치·참조 사진이 다 있는지 확인한다 (새 PC 구축 뒤 마지막 점검).

    python scripts/check_setup.py
빠진 것이 있으면 목록을 보여 주고 종료 코드 1. 받는 곳은 docs/[02]_구축_가이드.md.
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

WEIGHTS = {  # 경로: 대략 크기(바이트). 크기가 10% 넘게 다르면 받다 끊긴 것으로 본다
    "weights/sam/sam_ffhq_aging.pt": 2_270_547_237,
    "weights/sam/best_encoder.pt": 1_068_869_861,
    "weights/kin/gender_dir.pt": 77_737,
    "weights/ir_se50/model_ir_se50.pth": 175_367_323,
    "weights/facexlib/detection_Resnet50_Final.pth": 109_497_761,
    "weights/codeformer/codeformer_colorization.pth": 372_359_475,
    "weights/arcface/w600k_r50.onnx": 174_383_860,
}
HAIRFAST = [  # third_party/HairFastGAN/pretrained_models/ 아래 (추론에 쓰는 것만)
    "StyleGAN/ffhq.pt", "StyleGAN/ffhq_PCA.npz", "Rotate/rotate_best.pth", "Blending/checkpoint.pth",
    "PostProcess/pp_model.pth", "PostProcess/latent_avg.pt", "ArcFace/backbone_ir50.pth",
    "BiSeNet/face_parsing_79999_iter.pth", "BiSeNet/seg.pth",
    "FeatureStyleEncoder/143_enc.pth", "FeatureStyleEncoder/79999_iter.pth",
    "FeatureStyleEncoder/backbone.pth", "FeatureStyleEncoder/psp_ffhq_encode.pt",
    "encoder4editing/e4e_ffhq_encode.pt", "ShapeAdaptor/mask_generator.pth",
    "sean_checkpoints/CelebA-HQ_pretrained/latest_net_G.pth",
    "sean_checkpoints/CelebA-HQ_pretrained/latest_net_D.pth",
]
CODE = ["third_party/SAM/models/psp.py", "third_party/CodeFormer/basicsr/archs/codeformer_arch.py",
        "third_party/HairFastGAN/hair_swap.py"]
PATCHED = ["third_party/SAM/models/stylegan2/op/fused_act.py", "third_party/SAM/models/stylegan2/op/upfirdn2d.py",
           "third_party/HairFastGAN/models/stylegan2/op/fused_act.py",
           "third_party/HairFastGAN/models/stylegan2/op/upfirdn2d.py",
           "third_party/CodeFormer/basicsr/version.py"]


def main():
    bad = []
    for rel, size in WEIGHTS.items():
        p = ROOT / rel
        if not p.exists():
            bad.append(f"없음      {rel}")
        elif abs(p.stat().st_size - size) > 0.1 * size:
            bad.append(f"크기 다름 {rel} ({p.stat().st_size:,} / 예상 {size:,})")
    for f in HAIRFAST:
        p = ROOT / "third_party/HairFastGAN/pretrained_models" / f
        if not p.exists() or p.stat().st_size == 0:
            bad.append(f"없음      third_party/HairFastGAN/pretrained_models/{f}")
    for rel in CODE:
        if not (ROOT / rel).exists():
            bad.append(f"코드 없음 {rel} (git clone)")
    for rel in PATCHED:
        p = ROOT / rel
        if not p.exists() or (p.suffix == ".py" and "op" in p.parts and "Pure PyTorch" not in p.read_text(encoding="utf-8")):
            bad.append(f"패치 전   {rel} (python scripts/patch_third_party.py)")

    pr = yaml.safe_load((ROOT / "configs/hair_presets.yaml").read_text(encoding="utf-8"))
    for sex, styles in pr["styles"].items():
        for st in styles:
            names = (pr.get("selected_refs") or {}).get(f"{sex}/{st['key']}") or []
            d = ROOT / "data/hair_refs" / sex / st["key"]
            have = [n for n in names if (d / n).exists()] if names else list(d.glob("*.png")) if d.exists() else []
            if not have:
                bad.append(f"참조 없음 data/hair_refs/{sex}/{st['key']}/")

    try:
        import torch
        print(f"torch {torch.__version__}, CUDA {'사용 가능 ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '없음'}")
    except ImportError:
        bad.append("torch 가 설치되지 않음")
    if bad:
        print(f"빠진 것 {len(bad)}개:")
        for b in bad:
            print("  " + b)
        sys.exit(1)
    print("준비 완료: python -m app")


if __name__ == "__main__":
    main()
