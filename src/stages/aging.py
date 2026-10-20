"""4단계: 나이 변환. FFHQ 정렬 얼굴(RGB)과 목표 나이를 받아 변환된 얼굴(RGB)을 돌려준다."""
from argparse import Namespace

import cv2
import numpy as np

from ..paths import add_third_party, weight
from ..registry import register
from ._stub import stub


@register("aging", "sam")
class SAMAging:
    """yuval-alaluf/SAM. 입력 256x256, 출력 1024x1024, 목표 나이 0~100 연속값.

    SAM 의 W+ latent(18층) = pSp 재구성 latent(신원) + SAM 인코더 잔차(나이 변화).
    잔차에 곱할 세기를 층 묶음별로 받는다 (모두 1 = 원본 SAM, 모두 0 = 재구성만).
    아동 → 60세에서 관찰한 효과 (묶음 하나만 켰을 때):
      coarse 0~3층  거의 변화 없음
      medium 4~7층  주름·처짐·하관 넓어짐 등 노화 대부분, 신원 손실 대부분
      fine   8~17층 피부색·채도, 흰머리, 수염 그늘

    부모 사진 반영(src/kinship.py):
      mix         학습 없음. 부모도 같은 목표 나이로 encode → 층 묶음별로 섞기 → decode
      encode_kin  학습형 (src/kin_adapter.py). config 의 aging.adapter 체크포인트가 있을 때만
    aging.encoder_lora: pSp 재구성 인코더 LoRA (도메인 적응). 원본 SAM 결과도 이 인코더를 쓴다.
    aging.encoder_full: SAM 나이 인코더(4채널 입력) 전체를 튜닝한 체크포인트 {"encoder": state_dict, ...}.
      원본 나이 인코더를 통째로 바꾼다. 재구성 인코더·디코더는 그대로 (/sam 화면에서 원본과 비교).
    """

    GROUPS = {"coarse": slice(0, 4), "medium": slice(4, 8), "fine": slice(8, 18)}

    def __init__(self, device="cuda", adapter=None, encoder_lora=None, encoder_full=None, **_):
        import torch
        add_third_party("SAM")
        from models.psp import pSp

        from .. import kin_adapter as ka
        from ..paths import ROOT

        self.torch = torch
        self.device = device
        ckpt_path = str(weight("sam", "sam_ffhq_aging.pt"))
        opts = torch.load(ckpt_path, map_location="cpu")["opts"]
        opts.update(checkpoint_path=ckpt_path, device=device)
        self.net = pSp(Namespace(**opts)).to(device).eval()

        load = lambda p: torch.load(ROOT / p, map_location="cpu")  # noqa: E731  (config 경로는 프로젝트 기준)
        if encoder_full:  # LoRA·어댑터가 모듈을 감싸기 전에 바꿔야 키 이름이 맞는다
            full = torch.load(ROOT / encoder_full, map_location="cpu", weights_only=True)
            self.net.encoder.load_state_dict(full["encoder"])
            self.encoder_full_info = {k: full.get(k) for k in ("step", "metrics")}
            del full
        self.encoder_lora = encoder_lora and ka.load_into(load(encoder_lora), self.net, device)
        self.adapter, self.adapter_mode, self.idnet = None, None, None
        if adapter:
            ckpt = load(adapter)
            if ckpt.get("encoder_lora") != encoder_lora:
                print(f"[경고] 어댑터는 encoder_lora={ckpt.get('encoder_lora')} 캐시로 학습됨 (현재 {encoder_lora})")
            self.adapter, self.adapter_mode = ka.load_into(ckpt, self.net, device), ckpt["mode"]
            if self.adapter_mode == "encoder":  # 평소(원본 SAM 조건)에는 끄고 encode_kin 안에서만 켠다
                self.adapter.set_enabled(False)
            self.idnet = ka.IDNet().to(device)

    def __call__(self, face_rgb: np.ndarray, target_age: int, randomize_noise=False, **params) -> np.ndarray:
        return self.decode(self.encode(face_rgb, target_age, **params), randomize_noise)

    def _layer_scale(self, coarse, medium, fine, strength=1.0):
        scale = self.torch.zeros(1, 18, 1, device=self.device)
        for name, value in (("coarse", coarse), ("medium", medium), ("fine", fine)):
            scale[:, self.GROUPS[name]] = strength * value
        return scale

    def _input(self, face_rgb: np.ndarray, target_age: int):
        torch = self.torch
        x = cv2.resize(face_rgb, (256, 256), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(x).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5)
        age = torch.full((1, 256, 256), target_age / 100.0)  # SAM의 AgeTransformer와 동일
        return torch.cat([x, age]).unsqueeze(0).to(self.device)

    def latents(self, face_rgb: np.ndarray, target_age: int):
        """(pSp 재구성 latent, SAM 나이 잔차), 각각 (1, 18, 512)."""
        x = self._input(face_rgb, target_age)
        with self.torch.no_grad():
            base = self.net.pretrained_encoder(x[:, :3]) + self.net.latent_avg  # pSp.forward 와 동일
            residual = self.net.encoder(x)
        return base, residual

    def encode(self, face_rgb: np.ndarray, target_age: int, strength=1.0, coarse=1.0, medium=1.0,
               fine=1.0, sex=None, gender_strength=2.0, gender_dir=None, gender_refine=False, **_):
        """목표 나이의 W+ latent (1, 18, 512). sex('M'|'F')를 주면 성별 보정 (src/gender.py)."""
        base, residual = self.latents(face_rgb, target_age)
        return self._gender(base + self._layer_scale(coarse, medium, fine, strength) * residual, sex, gender_strength,
                            gender_dir, gender_refine)

    def _gender(self, w, sex, gender_strength, gender_dir=None, gender_refine=False):
        """결과가 고른 성별 쪽에서 벗어났을 때만 latent 의 성별 방향으로 되돌린다 (docs/[16], [28]).
        gender_dir: 방향 파일 (None = 첫 버전 0~7층). gender_refine: 방향 보정 뒤 그림을 보고 몇 번 더 고침 (IR-SE50)."""
        if sex not in ("M", "F") or not gender_strength:
            return w
        from ..gender import DEFAULT, GenderFix, GenderRefine
        path = gender_dir or DEFAULT
        if getattr(self, "gender_fix", None) is None or self._gender_path != path:
            self.gender_fix, self._gender_path, self.gender_refiner = GenderFix(path, device=self.device), path, None
        w2 = self.gender_fix(w, sex, gender_strength)
        if gender_refine and self.gender_fix.head is not None:
            if self.gender_refiner is None:
                if self.idnet is None:
                    from ..kin_adapter import IDNet
                    self.idnet = IDNet().to(self.device)
                kw = gender_refine if isinstance(gender_refine, dict) else {}
                self.gender_refiner = GenderRefine(self.net.decoder, self.idnet, self.gender_fix.head,
                                                   device=self.device, **kw)
            w2, _ = self.gender_refiner(w2, sex, w_orig=w)
        return w2

    # ---------------------------------------------------------------- 학습형 부모 반영

    def parent_feats(self, face_rgb: np.ndarray) -> dict:
        """부모 한 명의 조건: pSp 재구성 latent + IR-SE50 임베딩."""
        from ..kin_adapter import to_tensor256
        with self.torch.no_grad():
            w = self.latents(face_rgb, 0)[0]
            emb = self.idnet(to_tensor256(face_rgb)[None].to(self.device))
        return {"w": w, "emb": emb}

    def encode_kin(self, face_rgb: np.ndarray, target_age: int, parents: list, weights=None, **params):
        """parents = [parent_feats(...)] (빈 리스트 = 부모 없이 어댑터만). 반환: W+ (1, 18, 512).
        성별 보정(params 의 sex)은 어댑터를 거친 마지막 latent 에 한다."""
        torch = self.torch
        g = (params.pop("sex", None), params.pop("gender_strength", 2.0), params.pop("gender_dir", None),
             params.pop("gender_refine", False))
        from ..kin_adapter import make_cond
        avg = self.net.latent_avg
        if parents:
            wt = torch.tensor(weights or [1.0] * len(parents), device=self.device).view(-1, 1, 1)
            par_w = (torch.cat([p["w"] for p in parents]) * wt).sum(0, keepdim=True) / wt.sum()
            par_emb = torch.nn.functional.normalize((torch.cat([p["emb"] for p in parents]) * wt[:, 0]).sum(0, keepdim=True))
            mask = torch.ones(1, device=self.device)
        else:
            par_w, par_emb, mask = avg[None].clone(), torch.zeros(1, 512, device=self.device), torch.zeros(1, device=self.device)
        cond = make_cond(par_emb, par_w, torch.tensor([float(target_age)], device=self.device), mask, avg)
        with torch.no_grad():
            if self.adapter_mode == "latent":
                w = self.adapter(self.encode(face_rgb, target_age, **params), par_w, cond)[0]
                return self._gender(w, *g)
            self.adapter.set_enabled(True)  # encoder: SAM 나이 인코더 안의 LoRA + 부모 게이트
            self.adapter.set_cond(cond)
            try:
                return self._gender(self.encode(face_rgb, target_age, **params), *g)
            finally:
                self.adapter.clear_cond()
                self.adapter.set_enabled(False)

    def decode(self, codes, randomize_noise=False) -> np.ndarray:
        with self.torch.no_grad():
            out = self.net.decoder([codes], input_is_latent=True, randomize_noise=randomize_noise)[0][0]
        return out.clamp(-1, 1).add(1).div(2).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()

    def mix(self, child_codes, parent_codes, weights=None, coarse=0.5, medium=0.0, fine=0.0):
        """아이 latent 에 부모 latent(들의 가중 평균)를 층 묶음별 비율로 섞는다.
        비율 0 = 아이 그대로(원본 SAM), 1 = 그 층은 부모 것으로 교체."""
        weights = weights or [1.0] * len(parent_codes)
        parent = sum(w * c for w, c in zip(weights, parent_codes)) / sum(weights)
        alpha = self._layer_scale(coarse, medium, fine)
        return (1 - alpha) * child_codes + alpha * parent

    def reconstruct(self, face_rgb: np.ndarray, randomize_noise=False, **_) -> np.ndarray:
        """나이 변화 없이 pSp latent 만 디코딩. 입력과 안 닮으면 inversion 단계가 신원을 잃는 것."""
        return self(face_rgb, 0, strength=0.0, randomize_noise=randomize_noise)


stub("aging", "lats", "https://github.com/royorel/Lifespan_Age_Transformation_Synthesis")
stub("aging", "cusp", "https://github.com/guillermogotre/CUSP")
stub("aging", "fading", "https://github.com/MunchkinChen/FADING")
stub("aging", "hrfae", "https://github.com/InterDigitalInc/HRFAE")
