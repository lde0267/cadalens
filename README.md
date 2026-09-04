# CadaLens

지적도의 **등록 지목**과 최근 **정사영상의 실제 토지이용**을 필지 단위로 대조해
**무단 형질변경 의심 필지**를 스크리닝한다. 현재 범위는 **임야(산지전용)** 와 **농지(농지전용)**,
두 대상을 독립 파이프라인으로 설계한다.

파일럿 대상: 경기 안성시 도엽 `37714092` (정사영상 0.25 m/px, 연속지적도).

---

## 파이프라인

```
정사영상 TIF ─┐
             ├─▶ georeference ─▶ extract parcels ─▶ clip chips ─▶ RemoteCLIP zero-shot ─▶ 필지별 p_forest / 라벨
연속지적도 ───┘                                   (224×224)      (ViT-B/32)
```

1. **georeference** — 배포 과정에서 날아간 GeoTIFF 태그를 메타데이터 XML로 복원 (도엽번호/원점좌표 → CRS·transform, EPSG:5186)
2. **extract parcels** — 연속지적도 중 영상 범위에 완전히 들어오는 필지만 추출 (임야 428필지)
3. **clip chips** — 필지 폴리곤으로 정사영상을 잘라 224px 칩 생성. 칩 방식이 실험 변수:
   - `mask` — 폴리곤 밖 검정, 정사각 패딩
   - `context` — 넓은 창(이웃 포함) + 외곽선
   - `real` — 필지 bbox를 224에 꽉 채움, 밖도 원본 정사영상, alpha = 폴리곤 마스크(1/0) **← 현재 표준**
4. **classify** — RemoteCLIP(chendelong/RemoteCLIP, ViT-B/32) zero-shot.
   "임야다운" 문구군·"형질변경다운" 문구군과 이미지의 코사인 → 2-way softmax → `p_forest`.
   `p_forest < τ` → 형질변경의심.

---

## 현재 결과 (임야, 손라벨 150개 기준)

검증셋: 임야 428필지 중 면적 5분위 층화표집 150개 육안 라벨 → **형질변경 88 / 임야 62** (등록 임야의 59%가 실제 형질변경).

| 구성 | best τ | P | R | F1 |
|---|---|---|---|---|
| v1 CLS · v1 프롬프트 · mask | 0.95 | 0.75 | 0.84 | 0.79 |
| v1 CLS · v2 프롬프트 · mask | 0.80 | 0.70 | 0.84 | 0.76 |
| **v1 CLS · v2 프롬프트 · real** | **0.85** | **0.78** | **0.82** | **0.80** |
| A1 토큰드롭 · v2 프롬프트 · real | 0.95 | 0.61 | 0.94 | 0.74 |

**운영 권고**: `classify_binary.py --prompts v2` + `real` 칩 + `--tau 0.80~0.85`.

### 핵심 발견

- **τ(임계값)가 지배적 레버.** 기본 0.5는 재현율 0.2로 무의미. 운영점은 0.8~0.95
- **프롬프트(v1→v2) > 인코딩 변형.** A1 토큰드롭·C2 dense per-patch 는 whole-image CLS 대비 개선 없음/실패
- **재현율 천장 ~0.82.** 고대비 전환(관통도로·태양광·비닐하우스·건물·신선 나지)은 잡히나, 저대비 전환(벌채 후 재식생 풀밭·묘지·오래된 나지)은 단일 시점 겨울 RGB로 구분 불가 → **다시점 변화탐지 필요**

시도한 방법론 비교는 [`docs/experiments.md`](docs/experiments.md).

---

## 저장소 구조

```
scripts/
  georeference_from_metadata.py     정사영상 지오레퍼런싱
  extract_parcels_within_image.py   영상 완전포함 필지 추출
  clip_parcel_chips.py              필지 → 224px 칩 (mask / context / real)
  lib/remoteclip_backbone.py        RemoteCLIP 로드·프롬프트 임베딩·패치 유틸·A1 인코딩
  imya/
    classify_binary.py              whole-image CLS 2-way softmax (권장)
    classify_a1.py                  폴리곤 밖 패치토큰 드롭 (A1)
    classify_c2.py                  MaskCLIP dense per-patch + max/frac (C2, 실패)
  nongji/
    classify_farmland.py            전·답·과 형질변경 스크리닝
  labeling/
    make_label_set.py               손라벨용 미리보기 + 자체완결 HTML 도구 생성
    make_fit_previews.py            필지가 프레임에 꽉 차는 미리보기 재렌더
    finalize_labels.py              labels.json → labels.csv + 방법별 P/R/F1 채점
  legacy/run_remoteclip.py          초기 전지목 파일럿 (대체됨)
data/
  reference/     지목 코드·분류 테이블 (버전관리 O)
  labels/imya_eval_150/  손라벨 150개 (labels.csv, manifest.csv)
  (satellite/ cadastral/ chips/ remoteclip/ output/ 는 비버전 — data/README.md 참고)
```

---

## 설치 · 실행

```bash
pip install -r requirements.txt
# RemoteCLIP 가중치는 첫 실행 시 HuggingFace 에서 models/remoteclip/ 로 자동 캐시

# 1) 정사영상 지오레퍼런싱
python scripts/georeference_from_metadata.py

# 2) 영상 완전포함 필지 추출
python scripts/extract_parcels_within_image.py

# 3) 임야 칩 생성 (real 모드)
python scripts/clip_parcel_chips.py --jimok 임 --mode real --limit 0 --outdir data/chips/37714092_imya_real

# 4) 분류 (권장 구성)
python scripts/imya/classify_binary.py --cpu --prompts v2 \
    --chips data/chips/37714092_imya_real --tau 0.85

# 농지
python scripts/nongji/classify_farmland.py --cpu --jimok 전,답,과
```

### 검증셋 라벨링

```bash
python scripts/labeling/make_label_set.py        # → data/labels/imya_eval_150/label_tool.html
# 브라우저에서 label_tool.html 열고 1/2/3 키로 라벨 → "결과 저장" → labels.json
python scripts/labeling/finalize_labels.py        # 채점
```

---

## 데이터

원본 정사영상·지적도, 생성 칩, 실험 산출물은 용량 때문에 저장소에 포함하지 않는다.
필요한 파일과 배치 위치는 [`data/README.md`](data/README.md) 참고.

## 한계 / 다음

- 단일 시점 겨울 RGB → 저대비 전환 판별 한계. **연도별 정사영상 확보 시 변화탐지 레이어 추가**
- 프롬프트를 산지전용 유형별로 전문가(산림·측량·단속) 소견 반영해 재정의
- 검증셋 확대 및 "형질변경"의 조작적 정의 확정 (초지·과수·현황도로·자연 나지 처리)
