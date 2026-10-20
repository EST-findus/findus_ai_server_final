"""성별 보정: 나이 변환 결과가 고른 성별에서 벗어났을 때만 latent 를 그 성별 쪽으로 되돌린다.

1) GenderFix (방향): W+ latent 안의 '성별 방향'으로 민다. 학습 없이 train 사진의 pSp latent 로 구한 선형 분류기다
   (scripts/fit_gender_direction.py).
     s = <(w - latent_avg)[층], d>      d = 단위 벡터, 남자일수록 s 가 크다
     남자로 고정: s < lo_M 이면 lo_M 까지만 올린다 / 여자로 고정: s > hi_F 이면 hi_F 까지만 내린다
   이미 그 성별 쪽에 충분히 있으면 latent 를 건드리지 않는다 → 신원 손실이 없다.

2) GenderRefine (그림 보고 고치기): 방향 보정 뒤에도 디코딩한 얼굴이 고른 성별로 안 보이면, latent 를 몇 번 조금씩 고친다.
     손실 = 성별(IR-SE50 임베딩 위 로지스틱 판정기) + 신원(IR-SE50, 보정 전 결과와 같은 사람) + latent 변화량
   latent 방향 밖에 있는 성별 단서(수염 그늘·질감 등)까지 잡으려는 것. 판정기가 이미 그 성별로 보면 바로 멈춘다.
   학습용 망(IR-SE50)만 쓴다. 평가용(insightface genderage, CLIP, ArcFace)과 섞지 않는다.
"""
import torch
import torch.nn.functional as F

from .paths import ROOT

DEFAULT = "weights/kin/gender_dir.pt"


class GenderFix:
    def __init__(self, path=DEFAULT, device="cuda"):
        ck = torch.load(ROOT / path, map_location="cpu")
        self.d = ck["dir"].to(device)          # (18, 512), 쓰는 층 밖은 0, 전체 노름 1
        self.avg = ck["latent_avg"].to(device)
        self.bounds = ck["bounds"]             # {"M": 남자 쪽 하한, "F": 여자 쪽 상한}
        self.head = ck.get("head")             # IR-SE50 임베딩 성별 판정기 (GenderRefine 용)

    def score(self, w: torch.Tensor) -> torch.Tensor:
        """w (N,18,512) → (N,) 남자일수록 큼."""
        return ((w - self.avg) * self.d).sum((1, 2))

    def __call__(self, w: torch.Tensor, sex: str, strength: float = 1.0) -> torch.Tensor:
        """sex = 'M' | 'F'. strength 1 = 경계까지, 1.5 = 경계보다 0.5배 더."""
        if sex not in ("M", "F"):
            return w
        s = self.score(w)
        if sex == "M":
            shift = (self.bounds["M"] - s).clamp(min=0)
        else:
            shift = (self.bounds["F"] - s).clamp(max=0)
        return w + (strength * shift).view(-1, 1, 1) * self.d


class GenderRefine:
    """방향 보정한 latent 를 디코딩한 얼굴이 고른 성별로 보일 때까지 (최대 steps 번) latent 를 고친다.

    G = StyleGAN 디코더, idnet = kin_adapter.IDNet (IR-SE50), head = GenderFix.head.
    layers: 고칠 층 (0 ~ layers-1). 6GB 에 맞춰 batch 1, bf16 디코딩 (kin_adapter.decode_grad).
    기본값은 dev 비교로 골랐다 (docs/[28] 7장): lr 0.01·신원 1 은 신원·나이가 크게 흔들려서 lr 0.003·신원 3.
    """

    def __init__(self, G, idnet, head: dict, steps=40, lr=0.003, layers=12, target_p=0.8,
                 w_id=3.0, w_lat=0.5, device="cuda"):
        self.G, self.idnet, self.device = G, idnet, device
        self.G.requires_grad_(False)          # 고치는 것은 latent 뿐. 가중치 기울기는 메모리만 쓴다
        self.coef, self.bias = head["coef"].to(device), head["bias"]
        self.steps, self.lr, self.target_p, self.w_id, self.w_lat = steps, lr, target_p, w_id, w_lat
        self.mask = torch.zeros(1, 18, 1, device=device)
        self.mask[:, :layers] = 1

    def _decode(self, w):
        from .kin_adapter import decode_grad
        return decode_grad(self.G, w)

    def p_male(self, w: torch.Tensor) -> float:
        with torch.no_grad():
            return torch.sigmoid(self.idnet(self._decode(w)) @ self.coef + self.bias).item()

    def __call__(self, w: torch.Tensor, sex: str, w_orig: torch.Tensor = None):
        """w (1,18,512) 방향 보정한 latent, w_orig 보정 전 latent (신원 기준, 없으면 w). 반환 (latent, 정보 dict)."""
        if sex not in ("M", "F"):
            return w, {"steps": 0}
        with torch.enable_grad():  # 화면 쪽은 no_grad 안에서 부른다
            return self._run(w, sex, w_orig)

    def _run(self, w, sex, w_orig):
        sign = 1.0 if sex == "M" else -1.0
        w = w.detach()
        with torch.no_grad():
            ref = self.idnet(self._decode(w if w_orig is None else w_orig.detach()))
        delta = torch.zeros_like(w, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=self.lr)
        p0 = p = None
        for step in range(self.steps + 1):
            emb = self.idnet(self._decode(w + delta * self.mask))
            logit = emb[0] @ self.coef + self.bias
            p = torch.sigmoid(sign * logit).item()   # 고른 성별일 확률
            if p0 is None:
                p0 = p
            if p >= self.target_p or step == self.steps:
                break
            loss = (F.softplus(-sign * logit)
                    + self.w_id * (1 - F.cosine_similarity(emb, ref).mean())
                    + self.w_lat * (delta * self.mask).pow(2).sum(-1).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
        return (w + delta.detach() * self.mask), {"steps": step, "p_before": p0, "p_after": p}
