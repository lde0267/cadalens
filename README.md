# CadaLens

지적도의 **등록 지목**과 최근 **정사영상의 실제 토지이용**을 필지 단위로 대조해
**무단 형질변경 의심 필지**를 스크리닝한다. 대상은 **임야(산지전용)** 와 **농지(농지전용)**.

파일럿: 경기 안성시 도엽 `37714092` (정사영상 0.25 m/px, 연속지적도).

---

## 폴더 구조 — 3 단계

```
01_preprocess/     원본 → 칩 → 라벨.  독립적으로 돌아가는 데이터 준비 단계.
  raw/             원본 정사영상(tif+xml) · 연속지적도(shp)         [git 제외]
  derived/         s1·s2 산출: georef tif · 영상포함 필지 gpkg       [git 제외]
  chips/           s3 산출: <sheet>_<mode>/ (mask·context·real)      [git 제외]
  reference/       지목 코드·분류 테이블                             [버전관리]
  labels/          손라벨 세트 (labels.csv·manifest.csv·split.csv)   [버전관리]
  scripts/         s1_georeference → s2_extract_parcels → s3_make_chips  + labeling/

02_experiments/    이미지 처리 방식·프롬프트 실험. 결론은 docs/experiments.md.
  image_method/    classify_wholeimage · classify_interior · classify_densepatch  (인코딩 변형 프로브)
  prompt_ablation/ run_ablation.py · report.py · prompt_sets/*.json
  legacy/          run_remoteclip_alljimok.py · nongji_farmland_standalone.py
  results/         모든 실험 산출물                                  [git 제외]

03_pipeline/       전체 파이프라인. JSON 설정으로 이미지방식·프롬프트를 갈아끼운다.
  run.py           단계별 러너 (chips → classify → decide → eval)
  configs/*.json   프로파일 (imya_precision · imya_recall · imya_realfit · nongji)
  prompts/*.json   운영 프롬프트
  stages/          chips.py · classify.py · decide.py · evaluate.py
  runs/<config>/   실행별 단계 산출물 + config 동결 복사             [git 제외]

common/            공용 라이브러리 (버전관리)
  paths.py         저장소 표준 경로 (어디서 실행하든 절대경로 해결)
  geoenv.py        PROJ/GDAL 환경격리 (rasterio import 전에)
  remoteclip_backbone.py   모델 로드·텍스트 임베딩·패치 유틸·encode_kept(내부 패치만 통과)
  encoders.py      encode_whole(fill=) / encode_interior()  — 이미지 처리 방식 레버
  prompts.py       프롬프트 세트 로딩 (레거시 positive/negative + N그룹)
  scoring.py       τ 스윕 · confusion · P/R/F1 · 판정불가 밴드

docs/experiments.md   시도한 방법론 비교 (칩 방식 · 인코딩 · 프롬프트 · 채움)
```

---

## 설치

```bash
pip install -r requirements.txt
# RemoteCLIP 가중치는 첫 실행 시 HuggingFace 에서 models/remoteclip/ 로 자동 캐시
```

원본 데이터 배치는 [`01_preprocess/README.md`](01_preprocess/README.md).

---

## 1) 데이터 준비 (01_preprocess)

```bash
python 01_preprocess/scripts/s1_georeference.py        # → derived/*_georef.tif
python 01_preprocess/scripts/s2_extract_parcels.py     # → derived/*_parcels_within.gpkg  (임야 428)
python 01_preprocess/scripts/s3_make_chips.py --jimok 임 --mode mask --limit 0   # → chips/37714092_mask/
```

라벨링:

```bash
python 01_preprocess/scripts/labeling/make_label_set.py   # → labels/imya_eval_150/label_tool.html
# 브라우저로 label_tool.html 열고 1/2/3 키로 라벨 → "결과 저장" → labels.json
python 01_preprocess/scripts/labeling/make_split.py       # dev/test 50:50 (1회, seed 42)
python 01_preprocess/scripts/labeling/finalize_labels.py  # labels.json → labels.csv + 폴더 분류
```

## 2) 파이프라인 실행 (03_pipeline)

```bash
# 전체 (chips → classify → decide → eval)
python 03_pipeline/run.py --config 03_pipeline/configs/imya_precision.json --stage all --cpu

# 단계별로 — 중간 산출물 확인
python 03_pipeline/run.py --config 03_pipeline/configs/imya_precision.json --stage classify --cpu
#   → runs/imya_precision/2_scores.csv  를 열어보고
python 03_pipeline/run.py --config 03_pipeline/configs/imya_precision.json --stage decide --cpu
#   → runs/imya_precision/3_labels.csv · 3_ranked.png · 3_summary.json
python 03_pipeline/run.py --config 03_pipeline/configs/imya_precision.json --stage eval --cpu
#   → runs/imya_precision/4_eval.json  (손라벨 있으면 P/R/F1)
```

**이미지 처리 방식·프롬프트 교체** = config JSON 4줄만 수정:

```json
"image":    { "chip_mode": "mask", "fill": "mean", "encoder": "whole" },
"prompt_file": "03_pipeline/prompts/imya_forest.json",
"decision": { "mode": "binary", "tau": 0.93 }
```

- `chip_mode` ∈ {mask, context, real} · `fill` ∈ {black, mean, gray, inpaint}(mask 전용) · `encoder` ∈ {whole, interior}
- `decision.mode` = `binary`(임야) | `farmland`(농지: built_tau + 비닐하우스 side-flag)

### 프로파일 (검증셋 150 기준, docs/experiments.md)

| config | 구성 | P | R | F1 | 용도 |
|---|---|---|---|---|---|
| `imya_precision` | mask + mean-fill + whole + τ 0.93 | 0.85 | 0.76 | 0.80 | 정밀 판정 |
| `imya_recall` | interior + 한국산림 pos + τ 0.83 | 0.70 | 0.98 | 0.82 | 1차 고재현 스크리닝 |
| `imya_realfit` | real 칩 + v2 서술 + whole + τ 0.85 | 0.78 | 0.82 | 0.80 | real 칩 대안 |
| `nongji` | interior + farm/built + greenhouse flag | — | — | — | 농지 |

---

## 한계 / 다음

- 단일 시점 겨울 RGB → 저대비 전환(벌채 후 재식생·묘지·오래된 나지) 판별 한계. **연도별 정사영상 확보 시 변화탐지 레이어 추가.**
- 프롬프트를 산지전용 유형별로 전문가(산림·측량·단속) 소견 반영해 재정의.
- 검증셋 확대 및 "형질변경"의 조작적 정의 확정 (초지·과수·현황도로·자연 나지 처리).

> `checkoutput.qgz` 는 재구조화 전 경로를 참조하므로 레이어를 다시 연결해야 한다.
