# 01_preprocess — 데이터 준비

원본 정사영상·지적도를 넣으면 → 필지 칩 → 손라벨 세트까지. 파이프라인(`03_pipeline`)과
독립적으로 돌아간다. 각 단계가 자기 산출물을 남겨 중간 확인이 가능하다.

```
raw/        ← 여기에 원본을 넣는다        [git 제외]
derived/    s1·s2 산출                    [git 제외]
chips/      s3 산출                       [git 제외]
reference/  지목 테이블                   [버전관리]
labels/     손라벨 세트                   [버전관리: labels.csv·manifest.csv·split.csv·README]
scripts/    s1 → s2 → s3  +  labeling/
```

## 원본 배치 (raw/)

```
raw/
├── satellite/
│   ├── (B060)정사영상_2025_37714092.tif
│   └── (B060)정사영상메타데이터_202511001237714092.xml
└── cadastral/
    └── LSMD_CONT_LDREG_경기_안성시/          연속지적도 shp 세트 (.shp .dbf .prj .shx)
```

좌표계: 정사영상·지적도 모두 중부원점(EPSG:5186 / KGD2002 Central Belt 2010) 계열.
원본 tif 에 GeoTIFF 태그가 없으면 s1 이 메타데이터 XML 로 근사 복원한다(오차 ~1 m).

## 단계

| 스크립트 | 역할 | 산출 (확인 지점) |
|---|---|---|
| `scripts/s1_georeference.py` | 정사영상에 CRS·transform 주입 | `derived/(B060)…_georef.tif` |
| `scripts/s2_extract_parcels.py` | 영상 범위 완전포함 필지 추출 | `derived/anseong_37714092_parcels_within.gpkg` (임야 428) |
| `scripts/s3_make_chips.py` | 필지 폴리곤으로 224px 칩 생성 | `chips/<sheet>[_<group>]_<mode>/` (`index.csv` + `<pnu>_<jibun>.png` + `_montage.png`) |

```bash
python 01_preprocess/scripts/s1_georeference.py
python 01_preprocess/scripts/s2_extract_parcels.py

# 필지별 칩 — 임야 / 농지(전·답·과) 분리, 해당 지목 전 필지 (--limit 0)
python 01_preprocess/scripts/s3_make_chips.py --group imya   --mode real --limit 0
python 01_preprocess/scripts/s3_make_chips.py --group nongji --mode mask --limit 0
```

`--group imya` → 지목 `임` (428필지), `--group nongji` → `전·답·과` (약 2,513필지).
산출 폴더가 `chips/37714092_imya_<mode>/` · `chips/37714092_nongji_<mode>/` 로 갈려 섞이지 않는다.
단일 지목은 종전대로 `--jimok 임` (→ `chips/37714092_<mode>/`), 임의 조합은 `--jimok 전,답,과`.

칩 방식 (`--mode`, docs/experiments.md 참고):
- `mask` — 폴리곤 밖 검정, 정사각 패딩
- `context` — 넓은 창(이웃 포함) + 폴리곤 외곽선
- `real` — 필지 bbox 를 224 에 꽉 채움, 밖도 원본 영상, alpha=폴리곤(1/0)

`--outdir` 미지정 시 `chips/<sheet>[_<group>]_<mode>/` 로 자동. 파이프라인이 이 규칙으로 칩을 찾아 재사용한다.

## 라벨링 (labeling/)

| 스크립트 | 역할 |
|---|---|
| `make_label_set.py` | 면적 5분위 층화표집 + 미리보기 렌더 + 자체완결 라벨링 HTML 생성 |
| `make_fit_previews.py` | 필지가 프레임에 꽉 차도록 미리보기 재렌더 |
| `make_split.py` | 손라벨 150 → dev/test 50:50 층화. **1회만** (seed 42, 재생성 금지) |
| `finalize_labels.py` | `labels.json` → `labels.csv` (+ `excluded.csv`) + `임야/형질변경/보류/제외/` 폴더 분류 + (있으면) 파이프라인 결과 채점 |

```bash
python 01_preprocess/scripts/labeling/make_label_set.py
# labels/imya_eval_150/label_tool.html 을 브라우저로 열고 1/2/3/4 키 → "결과 저장" → labels.json
python 01_preprocess/scripts/labeling/finalize_labels.py
```

라벨 키: `1` 임야 · `2` 형질변경 · `3` 보류 · `4` 제외.
`제외` = 칩/영상 자체가 이상(구름·엉뚱한 필지·심한 왜곡)이라 평가 라벨로 못 쓰는 것 —
`finalize_labels.py` 가 `labels.csv` 에서 빼고 `excluded.csv` 로 따로 남긴다.

`label_tool.html` 은 이미지가 내장된 자체완결 파일이라 라벨링에는 이 파일 하나면 된다.
진행상황은 그 브라우저의 localStorage 에만 저장되므로 끝나면 반드시 "결과 저장" 을 눌러야 한다.
