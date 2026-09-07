# 02_experiments — 이미지 처리 방식 · 프롬프트 실험

칩 생성 방식과 모델 사용 방식을 단계적으로 바꿔가며 검증셋 150 개로 정량 평가한 기록.
**결론과 서술은 [`docs/experiments.md`](../docs/experiments.md)** 에 정리돼 있다.
여기에는 그 실험을 돌리는 코드만 있다. 산출물(`results/`)은 git 제외 — 필요하면 재실행한다.

## image_method/ — 인코딩 변형 프로브

| 스크립트 | 방식 | docs |
|---|---|---|
| `classify_wholeimage.py` | 전체 이미지 인코딩 (whole-image CLS, "v1") 2-way softmax. `--prompts {v1,v2}` / `--prompts-file` · `--fill {black,mean,gray,inpaint}` · `--tau` | §1·§5·§6d |
| `classify_interior.py` | 필지 내부만 인코딩 ("A1") — 폴리곤 밖 패치토큰 드롭. `--real` 시 alpha 무시 | §3·§6c |
| `classify_densepatch.py` | 패치별 dense 점수 ("C2") — MaskCLIP + max/frac 집계 (실패 기록) | §4 |

공통 백본·인코더·프롬프트 로딩은 `common/` 에서 가져온다.

```bash
python 02_experiments/image_method/classify_wholeimage.py --cpu --fill mean \
    --chips 01_preprocess/chips/37714092_mask \
    --out 02_experiments/results/image_method/wholeimage_mean.csv
```

## prompt_ablation/ — 프롬프트 세트 통제 비교

| 스크립트 | 역할 |
|---|---|
| `run_ablation.py` | 한 세트의 scores.csv × 손라벨 × split.csv → dev τ 스윕·test 1회 채점·판정불가밴드·안정성 → metrics.json (`common.scoring` 사용) |
| `report.py` | 세트별 metrics.json 을 모아 `results/prompt_ablation/summary.csv` + fig PNG |
| `prompt_sets/*.json` | set_a(v1) · set_b(v2) · set_c/c2/c_nowinter(mask 서술) · set_d/d2(산지전용 빈발) · set_e2/e3(context) |

```bash
S=set_b
python 02_experiments/image_method/classify_wholeimage.py --cpu \
    --chips 01_preprocess/chips/37714092_mask \
    --prompts-file 02_experiments/prompt_ablation/prompt_sets/$S.json \
    --out 02_experiments/results/prompt_ablation/$S/scores.csv --no-montage
python 02_experiments/prompt_ablation/run_ablation.py \
    --scores 02_experiments/results/prompt_ablation/$S/scores.csv \
    --out    02_experiments/results/prompt_ablation/$S/metrics.json
```

> 재구조화(2026-09) 때 기존 `set_*` 실험 산출물·fig 는 폐기했다. 위 절차로 재생성된다.
> 한국 산림특성 pos(구 set_e1)는 운영에 채택돼 `03_pipeline/prompts/imya_korean_forest.json` 로 이동.

## legacy/

| 스크립트 | 내용 |
|---|---|
| `run_remoteclip_alljimok.py` | 초기 전지목 파일럿 — visual_class 분류. `01_preprocess/reference/jimok_taxonomy.json` 기반 |
| `nongji_farmland_standalone.py` | 농지 farm/built + greenhouse. 진단 출력이 풍부한 독립 스크립트. 운영은 `03_pipeline/configs/nongji.json` |
