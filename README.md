# ai_server_final

FindUs 얼굴 변환 API 서버. `ai_lab` 의 모델 코드와 가중치를 쓴다.
나이 변환은 **SAM + 나이 인코더 전체 튜닝(`weights/sam/best_encoder.pt`) + 성별 보정**이다. 가족 데이터로 학습한 LoRA(latent 어댑터, 재구성 인코더 LoRA)는 쓰지 않는다.

| API | 하는 일 | 상태 |
|---|---|---|
| `POST /v1/aging` | 사진 + 나이 + 성별 → 그 나이 얼굴 | 동작 |
| `POST /v1/hair` | 나이 변환 결과(`result_id`) 또는 얼굴 사진 + 머리 스타일·색 → 머리 바꾼 얼굴 | 동작 |
| `POST /v1/glasses` | 결과 또는 사진 + 안경 스타일 → 안경 씌운 얼굴 | 명세만 (501) |

구성·명세·근거는 [docs/[00]_서버_구성_계획.md](docs/[00]_서버_구성_계획.md).

## 설치

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install --no-build-isolation --no-deps git+https://github.com/openai/CLIP.git

# ai_lab 에서 가중치(2.8GB)·외부 코드(6.1GB)·머리 참조 사진 복사 (git 에 없는 것들)
python scripts/sync_assets.py                # ai_lab 이 옆 폴더(../ai_lab)에 있을 때
python scripts/sync_assets.py --check        # 빠진 것만 확인
```

ai_lab 의 `.venv` 를 그대로 써도 된다: `..\ai_lab\.venv\Scripts\python -m app`

## 실행

```powershell
python -m app                                # 시험 페이지 http://127.0.0.1:8000/test/aging (나이 변환), /test/post (머리·화장·안경), 명세 /docs
$env:PORT = "8001"; python -m app            # 포트 바꾸기
```

GPU 가 6GB 하나라서 ai_lab 웹 서버(`web/server.py`)와 동시에 띄우면 메모리가 모자라 느려진다. 하나만 켠다.

| 환경 변수 | 기본 | 설명 |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `8000` | 외부에 열 때만 `0.0.0.0` |
| `API_KEY` | 없음 | 정하면 `X-API-Key` 헤더 필요 |
| `CORS_ORIGINS` | `*` | 쉼표로 여러 개 |
| `OUTPUT_DIR` | `outputs` | 결과 저장 위치 |
| `RESULT_TTL_HOURS` | `24` | 이보다 오래된 결과 삭제 (0 = 안 지움) |
| `QUEUE_TIMEOUT` | `300` | GPU 대기 최대 초 (넘으면 503) |
| `MAX_UPLOAD_MB` / `MAX_AGES` | `15` / `5` | 사진 크기 / 한 번에 만들 나이 개수 |

## 예시

```bash
# 1) 나이 변환
curl -X POST localhost:8000/v1/aging -F image=@kid.jpg -F ages=30,50 -F sex=M
# → {"result_id": "20261020-134315-e00dd3", "images": [{"key": "age30", "url": "/v1/results/.../images/age30.png"}, ...]}

# 2) 그 결과의 30세에 머리 (짧은 머리 + 애쉬 브라운)
curl -X POST localhost:8000/v1/hair -F result_id=20261020-134315-e00dd3 -F keys=age30 -F style=short -F color=ash_brown

# 나이 변환 없이 사진에 바로 머리
curl -X POST localhost:8000/v1/hair -F image=@face.png -F sex=F -F style=bob

# 고를 수 있는 스타일·색
curl localhost:8000/v1/hair/styles

# 이미지 받기
curl -o age30.png localhost:8000/v1/results/20261020-134315-e00dd3/images/age30.png
```

## 처리 시간 (RTX 1000 Ada 6GB, 사진 1장)

| 요청 | 시간 |
|---|---|
| 나이 변환, 나이 2개 | 13.0초 |
| 머리 스타일 + 색, 1장 (모델 처음 올릴 때) | 58.2~81.8초 |

## 폴더

| 폴더 | 내용 | git |
|---|---|---|
| `app/` | FastAPI 서버, `static/` 시험 페이지 (`aging.html` 나이 변환, `post.html` 후처리, 공통 `common.css`·`common.js`) | O |
| `src/` | ai_lab/src 모델 코드 (복사, 고치지 않음) | O |
| `configs/` | `final.yaml` (나이 변환 조합), `hair_presets.yaml` (ai_lab 과 같은 파일) | O |
| `scripts/` | `sync_assets.py` | O |
| `weights/`, `third_party/` | 가중치, SAM·CodeFormer·HairFastGAN | X |
| `data/hair_refs/` | 머리 참조 사진 (AI Hub, 재배포 금지) | X |
| `outputs/` | 요청 결과 (얼굴 사진) | X |

ai_lab 에서 모델 코드나 머리 목록을 바꾸면 `src/` 해당 파일과 `configs/hair_presets.yaml` 을 다시 복사하고 `python scripts/sync_assets.py` 를 한 번 더 돌린다. `configs/final.yaml` 은 서버용 조합(LoRA 없음, best_encoder)이라 ai_lab 파일로 덮어쓰지 않는다.
