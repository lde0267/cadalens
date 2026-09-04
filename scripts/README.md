# scripts/

## 공통 (데이터 준비 — 임야/농지 공유)

| 스크립트 | 역할 |
|---|---|
| `georeference_from_metadata.py` | 정사영상 TIF에 CRS·transform 주입 (EPSG:5186) |
| `extract_parcels_within_image.py` | 연속지적도 중 영상 범위 완전포함 필지 추출 |
| `clip_parcel_chips.py` | 필지 폴리곤으로 정사영상 잘라 224px 칩 생성 (`--mode mask` / `context` / `real`) |
| `lib/remoteclip_backbone.py` | RemoteCLIP 로드·프롬프트 임베딩(v1/v2)·패치 유틸·A1 토큰드롭 인코딩 |

`real` 모드 = 필지 bbox를 224에 꽉 채움, 폴리곤 밖도 원본 정사영상, alpha = 폴리곤 마스크(1/0). **현재 표준.**

## imya/ — 임야 파이프라인

| 스크립트 | 방식 |
|---|---|
| `classify_binary.py` | whole-image CLS 2-way softmax. **권장.** `--prompts {v1,v2}` `--tau` `--chips` |
| `classify_a1.py` | 폴리곤 밖 패치토큰 드롭 (A1). `--real` 시 alpha 무시하고 원본 RGB |
| `classify_c2.py` | MaskCLIP dense per-patch + max/frac (C2) |

검증셋 150개 기준: **v1 CLS + `--prompts v2` + `real` 칩 + `--tau 0.85` → P 0.78 / R 0.82 / F1 0.80** (최선).
A1·C2 는 whole-image CLS 대비 개선 없음. 상세는 `docs/experiments.md`.

```bash
python scripts/imya/classify_binary.py --cpu --prompts v2 \
    --chips data/chips/37714092_imya_real --tau 0.85
```

## nongji/ — 농지 파이프라인 (전·답·과)

| 스크립트 | 방식 |
|---|---|
| `classify_farmland.py` | A1 토큰드롭 + farm/built 2-way softmax. `p_built >= --built-tau` → 형질변경의심. 비닐하우스는 별도 플래그 |

```bash
python scripts/nongji/classify_farmland.py --cpu --jimok 전,답,과
```

## labeling/ — 검증셋 손라벨

| 스크립트 | 역할 |
|---|---|
| `make_label_set.py` | 층화표집 + 미리보기 렌더 + 자체완결 라벨링 HTML 생성 |
| `make_fit_previews.py` | 필지가 프레임에 꽉 차도록 미리보기 재렌더 |
| `finalize_labels.py` | `labels.json` → `labels.csv` + 방법별 P/R/F1 채점 |

```bash
python scripts/labeling/make_label_set.py
# label_tool.html 에서 1/2/3 키로 라벨 → "결과 저장" → labels.json → data/labels/imya_eval_150/
python scripts/labeling/finalize_labels.py
```

## legacy/

`run_remoteclip.py` — 초기 전지목 파일럿. imya/nongji 로 대체됨.
