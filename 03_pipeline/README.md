# 03_pipeline — 전체 파이프라인

JSON 설정 하나로 **이미지 처리 방식**(칩 mode · 폴리곤 밖 채움 · 인코더)과 **프롬프트**를
갈아끼우고, **단계별로 실행**해 중간 산출물을 확인한다.

```
run.py                러너
configs/*.json         프로파일
prompts/*.json         운영 프롬프트
stages/                chips · classify · decide · evaluate
runs/<config>/         실행별 산출물                 [git 제외]
```

## 실행

```bash
python 03_pipeline/run.py --config 03_pipeline/configs/imya_precision.json --stage all --cpu
python 03_pipeline/run.py --config 03_pipeline/configs/imya_precision.json --stage classify --cpu
python 03_pipeline/run.py --config 03_pipeline/configs/imya_precision.json --from decide --to eval --cpu
```

- `--stage {chips,classify,decide,eval,all}` 또는 `--from/--to` 범위. 기본 `all`.
- run 폴더 = `runs/<config 파일이름>/` (재실행 시 갱신). `--fresh` 면 타임스탬프 폴더.
- 각 단계는 이전 단계 산출을 run 폴더에서 읽는다. 없으면 `"먼저 --stage classify"` 로 멈춘다.

## 단계와 산출물

| 단계 | 하는 일 | 산출 (확인 지점) |
|---|---|---|
| `chips` | config 의 `{jimok, chip_mode}` 칩 폴더 확인, 없으면 `s3_make_chips.py` 호출 | `1_chips.txt` (칩 폴더 경로) |
| `classify` | 인코더 × 프롬프트 → 2-way softmax | `2_scores.csv` (필지별 `p_forest` / `p_screen` / closest_neg) |
| `decide` | τ → 라벨, 몽타주, 요약 | `3_labels.csv` · `3_ranked.png` · `3_summary.json` |
| `eval` | 손라벨 있으면 dev τ* 스윕 → test 채점 | `4_eval.json` (P/R/F1 · 판정불가밴드 · 안정성) |

`config.json` 은 run 폴더에 동결 복사된다.

## 설정 스키마

```json
{
  "target":   { "sheet": "37714092", "jimok": "임" },
  "image":    { "chip_mode": "mask", "fill": "mean", "encoder": "whole", "arch": "ViT-B-32" },
  "prompt_file": "03_pipeline/prompts/imya_forest.json",
  "decision": { "mode": "binary", "tau": 0.93 },
  "eval":     { "labels": "01_preprocess/labels/imya_eval_150/labels.csv",
                "split":  "01_preprocess/labels/imya_eval_150/split.csv" }
}
```

| 필드 | 값 |
|---|---|
| `image.chip_mode` | `mask` · `context` · `real` |
| `image.fill` | `black` · `mean` · `gray` · `inpaint`  (mask 칩에서만 효과, docs §6d) |
| `image.encoder` | `whole` (전체 이미지 CLS) · `interior` (필지 내부만 — 폴리곤 밖 패치토큰 드롭). `interior` 는 `keep_tau`·`min_keep` 추가 |
| `decision.mode` | `binary` (τ 는 `p_forest` 임계: `p_forest < tau → 형질변경의심`) |
| | `farmland` (`built_tau` 는 `p_screen` 임계 · `gh_min`·`gh_margin` 로 비닐하우스 side-flag) |

## 프롬프트 파일

두 형식 모두 로드된다 (`common/prompts.py`):

```json
{ "positive": ["...임야다움..."], "negative": ["...형질변경다움..."] }        # 레거시 2그룹
```
```json
{ "groups": { "forest": [...], "changed": [...], "greenhouse": [...] },        # N그룹
  "softmax": ["forest", "changed"], "positive_group": "changed",
  "side_flags": ["greenhouse"] }
```

`positive_group` = "형질변경 의심" 쪽 그룹.

## 프로파일

| config | image | prompt | decision | 검증셋 P/R/F1 |
|---|---|---|---|---|
| `imya_precision` | mask · mean · whole | imya_forest | τ 0.93 | 0.85 / 0.76 / 0.80 |
| `imya_recall` | mask · interior | imya_korean_forest | τ 0.83 | 0.70 / 0.98 / 0.82 |
| `imya_realfit` | real · whole | imya_realfit_v2 | τ 0.85 | 0.78 / 0.82 / 0.80 |
| `nongji` | mask · interior | nongji_farm_built | farmland built_tau 0.60 | — |
