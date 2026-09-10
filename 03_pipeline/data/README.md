# 03_pipeline/data — 입력 데이터

파이프라인은 이 폴더에 묶여 있지 않다 — `configs/*.json` 의 `target.chips` (또는 `run.py --chips`)
를 다른 칩 폴더로 바꾸면 그대로 돈다. png·gpkg·labels.json·index.csv 는 `.gitignore` (재현 가능),
`_make_*.py` 와 이 README 만 버전관리.

## 스모크 샘플 — `_make_sample.py`

세종 **36710022** 도엽(개발제한구역 검증용)에서 소수 추출.

| 폴더 | 원본 | 내용 |
|---|---|---|
| `imya_sample/` | `01_preprocess/chips/36710022_imya_real` | 임야 40 (GB 밖 20 + 안 20) |
| `nongji_sample/` | `01_preprocess/chips/36710022_nongji_mask` | 농지 45 (전·답·과) |
| `sejong_36710022_parcels.gpkg` | `01_preprocess/derived/sejong_36710022_parcels_within.gpkg` | 심각도 A(개발제한구역 저촉)용 필지 폴리곤 |

```bash
python 03_pipeline/data/_make_sample.py
```

## 지역별 검증 세트 — `_make_regions.py`

`01_preprocess/labels/labels/labels_<sheet>_<group>.json` (label_tool 손라벨) 마다 `<sheet>_<group>/`:

```
<sheet>_<group>/
  index.csv      원본 칩 index 스키마, 라벨된 필지만 (PNU 로 원본 index 에서 해결)
  <pnu>_*.png    라벨된 필지 칩 사본 ('제외' 라벨은 png 복사 안 함 — 채점 미사용)
  labels.json    원본 라벨 JSON 그대로 (라벨값: 형질변경 · 임야/농지 · 제외)
```

도엽: imya 7 (35604085·36709030·36710022·36813045·37709030·37714092·37813048),
nongji 6 (36710022 제외 동일). 조인 키 = PNU.

```bash
python 03_pipeline/data/_make_regions.py            # 전체
python 03_pipeline/data/_make_regions.py 36709030   # 특정 도엽
python 03_pipeline/eval_regions.py                  # 도엽별 채점
```
