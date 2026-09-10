# 03_pipeline — 무단 형질변경 우선순위 큐 (확정본)

**필지 칩을 넣으면, 현장확인 우선순위 순으로 정렬된 필지 리스트를 뽑는다.**
형질변경 O/X 자동판정이 **아니다** — zero-shot RemoteCLIP + 단일시점 RGB 로는 F1 ~0.6 이 천장이고
그 임계값이 도엽마다 흔들린다는 것을 13개 도엽 검증으로 확인했다. 대신 안정적인 것은 **순위(AUC ~0.85)** 이므로,
산출을 "이 필지들 먼저 보라" 는 **우선순위 큐 + 심각도** 로 정의한다.

임야(산지전용)와 농지(전·답·과, 농지전용)는 프롬프트·인코더·심각도 파라미터가 달라 **분리된 두 파이프라인**이다.
risk 점수 스케일도 다르므로 (임야 `p_forest` / 농지 `p_screen`) 큐는 합치지 않는다.

---

## 입력 (INPUT)

한 번 실행 = **1 도엽 × 1 지목군**. 필요한 것:

| # | 무엇 | 형식 | 필수 | 쓰이는 곳 |
|---|---|---|:--:|---|
| 1 | **필지 칩 폴더** | `<pnu>_<jibun>.png` (RGBA, alpha=폴리곤 마스크) + `index.csv` (chip,pnu,jibun,jimok,area_m2,valid_ratio,…) | ✅ | classify |
| 2 | **프롬프트 세트** | `prompts/*.json` — forest/changed (임야) · farm/built (+greenhouse) (농지) | ✅ | classify |
| 3 | **필지 폴리곤** | `.gpkg`, `PNU` 컬럼 | 심각도용 | severity (개발제한구역 교차·면적) |
| 4 | **개발제한구역 폴리곤** | `.shp` 또는 폴더 | GB 티어용 | severity |
| 5 | **검증 손라벨** | label_tool `.json` 또는 `pnu,label` csv | 검증만 | eval |

- 칩 폴더는 `s3_make_chips.py --mode {real,mask}` 산출과 같은 규약. `01_preprocess` 는 필요 없다.
- 3·4 가 없어도 파이프라인은 돈다 — 그 경우 심각도는 개발제한구역·면적 항 없이 `change_frac` 만으로, 티어는 T2/T3 만.

---

## 출력 (OUTPUT)

`runs/<name>/` 에 단계별 산출. **최종 산출물은 `5_worklist.csv`.**

### `5_worklist.csv` — 우선순위 큐 (전 필지)

| 컬럼 | 뜻 |
|---|---|
| `rank` | 최종 순위 (1 = 먼저 볼 것) |
| `tier` | `T1` 개발제한구역 저촉 · `T2` 대면적/고전환 · `T3` 나머지 |
| `risk` | 형질변경일 가능성 0~1 (`1-p_forest` 또는 `p_screen`) |
| `pnu` `jibun` `jimok` `area_m2` | 필지 식별·면적 |
| `closest_neg` | 추정 변화유형 (가장 가까운 형질변경 프롬프트 문구) |
| `change_frac` | 필지 내부 패치 중 형질변경으로 보이는 비율 (커버리지 가중) |
| `change_area_m2` | `change_frac × area_m2` — 추정 전환 면적 |
| `in_greenbelt` `gb_ratio` | 개발제한구역 저촉 여부·비율 |
| `severity_score` (0~100) `severity_level` (낮음/중간/높음) | 심각도 (순위엔 안 쓰임, 표시용) |
| `p_forest` `p_screen` `chip` | 원점수·칩 파일명 |

### 부가 산출물

| 파일 | 내용 |
|---|---|
| `2_scores.csv` | 필지별 RemoteCLIP 점수 — `p_forest`/`p_screen`, `change_frac`, 서브프롬프트별 코사인 16개, `closest_neg` |
| `3_labels.csv` · `3_ranked.png` · `3_summary.json` | 임계값 하나로 자른 **개략** O/X + 몽타주 (참고용, 신뢰 낮음) |
| `4_eval.json` | (라벨 있을 때) `ranking` 블록 — AUROC/Gini/AP · recall@{5,10,20,30,50}% · lift · precision@top · 예산→{80,90}%회수 |
| `4_cap.png` | CAP 곡선 (검토 예산 vs 위반 회수율) |
| `5_worklist.json` | 티어·등급 분포 요약 |
| `5_worklist.png` | rank 상위 몽타주 (캡션 = tier·risk·severity·GB) |

### 다중 도엽 병합 (`sejong_run.py` 방식)

지목군별로 여러 도엽 `5_worklist.csv` 를 이어붙여 **group 내 (tier, -risk) 재정렬** →
`runs/sejong_<group>_worklist.csv` (+ `sheet` 컬럼). `runs/sejong_summary.json` 에 티어별 실제 형질변경 정밀도.

---

## 파이프라인 단계

```
chips → classify → decide → severity → eval(선택)
```

| 단계 | 하는 일 | 산출 |
|---|---|---|
| `chips` | `target.chips` 폴더·지목 확인 | `1_chips.txt` |
| `classify` | RemoteCLIP zero-shot: 2-way softmax(forest/changed) + 패치별 dense 형질변경비율(`change_frac`) + 서브프롬프트 코사인 | `2_scores.csv` |
| `decide` | 임계값 하나로 개략 O/X + 몽타주 (참고용) | `3_labels.csv` · `3_ranked.png` |
| `severity` | **전 필지**에 risk + 연속 심각도 산출, 티어 배정, (tier, -risk) 정렬 | `5_worklist.csv` · `.json` · `.png` |
| `eval` | 손라벨 있으면 순위 지표(recall@budget 등) | `4_eval.json` · `4_cap.png` |

---

## 우선순위 = 티어 + risk (사전식)

두 축을 **하나로 섞지 않는다.** 정렬키 `(tier, -risk)` — tier 가 1순위, risk 는 같은 tier 안에서만.

**1단계 — tier 배정** (심각도의 *구성요소* 를 직접 사용, 연속 `severity_score` 아님):

| tier | 조건 |
|---|---|
| **T1** | `in_greenbelt` **AND** `risk ≥ risk_floor`(기본 0.5) |
| **T2** | (T1 아님) **AND** `risk ≥ risk_floor` **AND** (`area_m2 ≥ area_big_m2` **OR** `change_frac ≥ cf_hi`) |
| **T3** | 나머지 전부 |

**2단계** — 각 tier 안에서 `risk` 내림차순.
**3단계** — `T1 + T2 + T3` 이어붙여 `rank` 1..N.

함의: T2(risk 0.55) > T3(risk 0.99). 개발제한구역인데 깨끗해 보이면(risk < floor) T3 바닥으로.
"개발제한구역 위반 먼저 → 대규모 전환 → 나머지는 가능성순" — 감사·소명에 그대로 설명 가능.

## 심각도 점수 (0~100, 표시 전용)

```
severity_score = w_gb · [개발제한구역 저촉]
               + w_cf · min(change_frac / cf_ref, 1)
               + w_area · clip( (log(전환면적) − log(area_lo)) / (log(area_ref) − log(area_lo)), 0, 1 )
전환면적 = change_frac × area_m2
severity_level: < score_bands[0] 낮음 · < score_bands[1] 중간 · 이상 높음
```

기본: `w_gb 40 · w_cf 30 · w_area 30 · cf_ref 0.5 · area_lo 1000㎡ · area_ref 20000㎡(임야)/5000㎡(농지) · score_bands [30, 58]`.
개발제한구역 폴리곤 없으면 GB 항 0 → 그 도엽 심각도는 사실상 "전환 증거" 만 반영.

---

## 설정 스키마

```json
{
  "target":   { "chips": "data/36710022_imya", "jimok": "임" },
  "image":    { "chip_mode": "real", "fill": "black", "encoder": "whole", "arch": "ViT-B-32" },
  "prompt_file": "03_pipeline/prompts/imya_change.json",
  "decision": { "mode": "binary", "norm": "none", "tau": 0.85, "frac_tau": 0.22,
                "normal_label": "임야", "positive_label": "형질변경의심" },
  "severity": { "parcels": "data/sejong_36710022_parcels.gpkg",
                "greenbelt": "greenbelt/LSMD_CONT_UD801_세종",
                "risk_floor": 0.5, "area_big_m2": 30000, "cf_hi": 0.35,
                "w_gb": 40, "w_cf": 30, "w_area": 30,
                "cf_ref": 0.5, "area_lo_m2": 1000, "area_ref_m2": 20000, "score_bands": [30, 58] },
  "eval": {}
}
```

| 필드 | 값 |
|---|---|
| `target.chips` | 입력 칩 폴더 (`03_pipeline/` → 저장소 루트 → 절대 순으로 해결). `--chips` 로 오버라이드 |
| `target.jimok` | 대상 지목. 콤마 목록 (`전,답,과`) |
| `image.chip_mode` | `real` (임야, 폴리곤 밖도 실영상) · `mask` (농지, 밖 검정) |
| `image.encoder` | `whole` (전체 CLS) · `interior` (폴리곤 밖 패치토큰 드롭 — `keep_tau`·`min_keep`) |
| `image.patch_thr` · `last_mlp` | dense `change_frac` 계산 파라미터 (기본 0.5 / false) |
| `decision.mode` | `binary` (임야) · `farmland` (농지 — farm/built + greenhouse side-flag) |
| `decision.norm` · `norm_tau` · `frac_tau` | 개략 O/X 용 (참고 단계). 실측상 이득 없어 `none` 고정 권장 |
| `severity.*` | 위 "심각도 점수" 참조. 전부 선택, 기본값 있음 |
| `eval.labels` (`.json`/`.csv`) · `eval.split` | 있으면 순위 채점. `--labels` 로 오버라이드 |

**CLI 오버라이드** (`run.py`): `--chips` `--labels` `--severity-parcels` `--severity-greenbelt`(`none`=비활성) `--run-name` `--no-severity` `--cpu` `--batch`.

---

## 실행

```bash
# 단일 도엽 × 지목군
python 03_pipeline/run.py --config 03_pipeline/configs/imya.json \
  --chips data/36710022_imya --run-name 36710022_imya \
  --severity-parcels data/sejong_36710022_parcels.gpkg \
  --severity-greenbelt greenbelt/LSMD_CONT_UD801_세종 --stage all

# 세종시 전량 (2 도엽 × 임야/농지 = 4 run → 큐 2개)
python 03_pipeline/data/_make_regions.py      # data/<sheet>_<group>/ 준비 (라벨 도엽)
python 03_pipeline/sejong_run.py              # 4 run + 병합
python 03_pipeline/sejong_run.py --severity-only   # classify 재사용, 심각도 파라미터만 재조정

# 도엽별 순위 검증 (라벨 있는 도엽)
python 03_pipeline/eval_regions.py            # 13 도엽 recall@budget → runs/_regions_eval.csv
```

---

## 검증 결과 (세종 실측, 손라벨 대비)

**임야 542필지 (실제 형질변경 24%)** — 티어별 실제 위반 정밀도:

| tier | 필지 | 실제 위반 | 정밀도 |
|---|--:|--:|--:|
| T1 (개발제한구역) | 21 | 16 | **0.76** |
| T2 (대면적/고전환) | 36 | 31 | **0.86** |
| T3 | 485 | 84 | 0.17 |

- **상위 57필지(T1+T2) 중 47이 실제 위반 = 82%.**
- 티어정렬 recall@budget: 상위 10% → 34% · 20% → 56% · 30% → 69% · 50% → 88% 회수.

**농지 라벨분 373 (위반 18%)**: T2 정밀도 0.57 · T3 0.08. (T1 은 개발제한구역 도엽에 손라벨 없어 미검증.)

**도엽별 순위(zero-shot, 임계값 무관) 13도엽 요약**: 임야 pooled AUC 0.84 · rec@20% 0.49 (무작위 대비 2.5×) · 예산→위반 80% = 42%.
농지 AUC 0.84 · rec@20% 0.65 (3.2×) · 37%. 위반율 낮은 도엽일수록 lift 큼 (안성 임야 위반 5% → 예산 13% 로 위반 80% 회수).

---

## 알아둘 한계

1. **F1 천장 ~0.6.** 프롬프트 재설계(v1/v2/v3), z-정규화, LORO 선형프로브 전부 순위(AUC)를 못 올림 — 클래스 분포가 심하게 겹침(d-prime ~1.5). 이건 임계값이 아니라 신호의 문제.
2. **T3 에 위반이 남는다.** 임야는 전체 위반의 ~64% 가 T3. T1+T2 만 처리하면 놓치므로, T3 도 `risk` 순으로 예산 닿는 데까지 훑어야 한다.
3. **recall 80→90% 구간 비용 급증** (예산 임야 42→51%). 저대비 전환(묘지·벌채→풀)은 단일시점 RGB 로 순위가 안 매겨짐 → 다시점 신호 필요.
4. **개발제한구역 폴리곤은 지역 한정.** 세종은 금남면(36710022)만 있고 36709030 은 없음 → 후자엔 T1 없음, 심각도 "높음" 도 거의 없음.
5. **개략 O/X (`3_labels.csv`) 는 참고용.** τ 가 도엽마다 0.5~0.95 로 흔들려 배포 불가. 큐(`5_worklist.csv`)를 쓴다.

## 다음 레버 (순위 자체를 올리려면)

- **다시점 self-difference** — 같은 필지 두 시점 임베딩 차이. 계절·지역 드리프트 면역. (2차 시점 정사영상 필요)
- **비-CLIP 피처** — NDVI·텍스처·엣지밀도를 선형프로브에 결합.
- **세그멘테이션 + 면적비율** — 부분 전환을 조각 단위로.

---

## 파일 맵

```
run.py                러너 (단일 실행)
sejong_run.py         세종 4-run + 병합
eval_regions.py       도엽별 순위 검증 일괄
probe_loro.py         LORO 선형프로브 (실험 — 미채택)
configs/
  imya.json  nongji.json      확정 프로파일 2개
  variants/                    대체 이미지 처리 레시피 (참고)
prompts/
  imya_change.json  nongji_change.json    확정 (실제 산지전용·농지전용 유형 기반)
  imya_*.json                             과거 실험본
stages/  chips · classify · decide · severity · evaluate
data/                입력 (칩 폴더 · 필지 gpkg)              [png·gpkg·labels 는 git 제외]
greenbelt/           개발제한구역 폴리곤 (세종)
runs/<name>/         실행별 산출물                            [git 제외]
```

공용 라이브러리는 `common/` (RemoteCLIP 백본 · 인코더 · 프롬프트 로더 · 채점·정규화).
