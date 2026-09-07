# 프롬프트 ablation — 임야 · mask 칩

통제 실험: 칩·모델·인코딩·평가 프로토콜 전부 고정, **프롬프트 세트만 교체**.
데이터 428 (지목 임, mask 모드) · 손라벨 150 (형질변경 88 / 임야 62) ·
dev/test = 75/75 (면적 5분위 × 라벨 층화, seed 42, `01_preprocess/labels/imya_eval_150/split.csv`).
프로토콜: dev 에서 τ 0.50→0.99 스윕 → F1 최대 평탄대의 **중앙 τ** 를 τ\* 로 잡고 (dev 노이즈 과적합 완화),
test 75 는 그 τ\* 로 **1회만** 채점. positive = 형질변경, `p_forest < τ` → 형질변경의심.

## 결과 (summary.csv)

| set | pos/neg | dev τ\* | dev F1 | test P | test R | **test F1** | ΔF1 vs A | test 자체최적 F1 | 판정불가밴드(±0.05, 428) | 안정성 |
|---|---|---|---|---|---|---|---|---|---|---|
| **A · v1** 일반명사 (baseline) | 7/14 | 0.95 | 0.816 | 0.773 | 0.756 | **0.764** | — | 0.775 | 51.2 % | 안정 |
| B · v2 서술형 (real-fit용) | 6/7 | 0.79 | 0.771 | 0.720 | 0.800 | **0.758** | −0.006 | 0.776 | 19.4 % | 안정 |
| C · mask특화 서술형 (신규) | 6/12 | 0.96 | 0.824 | 0.679 | 0.844 | **0.752** | −0.012 | 0.787 | 35.3 % | 안정 |
| D · 산지전용 빈발유형 (도로 포함) | 6/9 | 0.99 | 0.804 | 0.643 | 0.800 | **0.713** | −0.051 | 0.761 | 43.0 % | 안정 |
| D2 · D 자연어꼬리 제거 | 6/9 | 0.99 | 0.800 | 0.706 | 0.800 | **0.750** | −0.014 | 0.750 | 54.7 % | 안정 |
| (진단) C2 · C −초지 +온실완화 | 6/11 | 0.98 | 0.824 | 0.661 | 0.822 | 0.733 | −0.031 | 0.791 | 42.3 % | 안정 |
| (진단) C-nowinter · C2 pos −겨울 | 6/11 | 0.98 | 0.800 | 0.698 | 0.822 | 0.755 | −0.009 | 0.764 | 52.1 % | 안정 |

**C2 −winter**: pos 에서 leafless/bare/winter 제거 → 자체최적 0.764, 판정불가밴드 52.1 % (C2 +10 pt).
→ 겨울 나목 명시가 분리도를 넓힌다 (B-1 C-winter 검증질문: 예).

**Set D — negative 를 실제 임야 산지전용 빈발유형으로 (전국 산지전용허가 기준 + 안성 데이터):**
태양광·전원주택 부지·공장창고·**도로(오탐 최다, 첫 항목 유지)**·개간(밭·과수)·조성나지·비닐하우스·묘지·야적/토취.
`prompt_sets/set_d.json`. **test F1 0.713 — 최악. A 대비 −0.051.**

- 실패 양상: 서술을 "실제 사례"로 자연스럽게 쓰니 (`"road cut across the forested slope"`,
  `"grassy burial mounds"`, `"panels on sloping ground"`) 음성문구가 **다시 겨울 임야를 흡인**.
  dev FP 18 중 도로 7 · 묘지 6. C 의 좁힌 도로문구(`"...cutting across the hillside"`, 도로 attractor 2)
  를 D 에서 `"...cut across the forested slope"` 로 바꾼 순간 도로 attractor 82, dev FP 7 로 회귀.
- **D2** (`prompt_sets/set_d2.json`): D 에서 `forested slope / sloping ground / grassy mounds / on former
  forest land` 같은 자연어·숲맥락 꼬리말 제거, 도로문구는 C 원문 복원. → test F1 0.750, 도로 attractor
  82→3, **그러나 판정불가밴드 54.7 % (전 세트 최악)**. p_forest 가 과압축돼 대부분 필지가 τ 바로 옆에 뭉침.
- 결론: negative 를 "실제 빈발유형"으로 좁혀도 F1 안 오름. 자연어로 쓰면 임야 흡인(FP), 평평하게 쓰면
  분리 붕괴(회색지대). 둘 다 v1 의 0.764 를 못 넘음.

C-tmpl (템플릿 앙상블 vs 단일) 은 지시서 A-2 (템플릿 세트 고정) 와 충돌하므로 실행하지 않음.

## Set E — context 칩 + 한국 산림특성 positive (별도 실험)

칩만 `01_preprocess/chips/37714092_context` (필지 중심 넓은 창 + 실장면 + 폴리곤 외곽선, 필지가 프레임의 **중앙값 14 %**)
로 바꾸고 나머지 프로토콜 동일. positive 를 **우리나라 산림특성**으로 재작성:
급경사 산지 / 침엽(소나무·잣·낙엽송) + 낙엽활엽(참나무류) 혼효림 / 겨울 갈색 참나무 + 상록 소나무 /
조림지 줄심기 / 임도·방화선 허용. (`03_pipeline/prompts/imya_korean_forest.json`(구 set_e1))

| set | 구성 | dev τ\* | test P | test R | **test F1** | ΔF1 vs A | 자체최적 F1 | 밴드(428) | 안정성 |
|---|---|---|---|---|---|---|---|---|---|
| E0 | context + v1 (대조) | 0.88 | 0.686 | 0.778 | 0.729 | −0.035 | 0.766 | 10.8 % | 안정 |
| E1 | context + 한국산림 pos + 산지전용 neg | 0.61 | 0.621 | 0.911 | 0.739 | −0.025 | 0.757 | 6.5 % | **불안정 (Δτ 0.28)** |
| E2 | context + 일반 pos + 산지전용 neg | 0.89 | 0.633 | 0.844 | 0.724 | −0.040 | 0.750 | 13.3 % | 안정 |
| **E3** | **context + 한국산림 pos + v1 넓은 neg** | 0.59 | **0.745** | **0.844** | **0.792** | **+0.028** | **0.792** | **5.1 %** | 안정 |

- **한국 산림특성 positive 가 context 에서 실효.** neg 를 v1 으로 고정하고 pos 만 일반→한국산림 으로
  바꾸면 **E0 0.729 → E3 0.792 (+0.063)**. mask 칩에서 겨울/한국형 pos 가 안 먹혔던 것과 대조 —
  context 는 프레임의 86 %가 주변 산이라, "steep forested Korean mountain, mixed pine and brown oak"
  같은 pos 가 실제 지배 장면과 정합해 마을 인접 임야도 p_forest 를 지켜냄.
- **"산지전용 빈발유형" neg (E1·E2) 는 context 에서 역효과.** `"bare soil ... where forest was logged"`,
  `"houses ... cut into the hillside"` 가 프레임 안 이웃(도로·나지·집)에 그대로 걸려 337/428 를 flag,
  τ 를 0.61 로 끌어내리고 E1 을 불안정하게 만듦.
- **E3 의 강점은 τ 비민감성.** test F1 이 τ 0.50~0.94 전 구간에서 **0.75~0.79** (mask A 는 τ 0.95
  근처에서만 0.78, 빗나가면 0.72). 운영상 τ 튜닝 부담이 사라짐. 판정불가밴드도 5 %.
- **단, 정밀도는 여전히 0.75** (428 중 61 % flag). 고재현(0.84)·저정밀 **1차 스크리닝**용으로 적합,
  정밀 판정 도구로는 부족. dev-fit τ\*=0.59 가 평탄대의 유리한 끝에 떨어진 점도 감안 (τ=0.72 였다면 F1 0.76).
- **판정**: A-3 +0.02 기준을 넘긴 **유일한 세트**. 다만 mask/context 는 칩이 달라 A 와 직접 대체 비교가
  아님 → "context + 한국산림 pos" 를 **고재현 스크리닝 프로파일**로 별도 채택 후보. `03_pipeline/prompts/imya_korean_forest.json`(구 set_e1)
  (E3 용도로는 neg 를 v1 으로: `prompt_sets/set_e3.json`).

## Set A1 — 모델이 필지 패치만 보게 (토큰 드롭)

`classify_interior.py` : patchify 후 폴리곤 커버리지 ≥ τ 인 패치토큰 + CLS 만 transformer 통과.
검정 패딩도, 주변 맥락도 임베딩에 안 들어감. mask 칩 (`01_preprocess/chips/37714092_mask`), 유지패치 중앙값 16/49.

| set | dev τ\* | test P | test R | **test F1** | ΔvsA | 자체최적 F1 | 밴드(428) | 안정성 |
|---|---|---|---|---|---|---|---|---|
| A1 + v1 | 0.98 | 0.688 | 0.733 | 0.710 | −0.054 | 0.734 | 56.8 % | 안정 |
| A1 + C (mask 서술) | 0.96 | 0.750 | 0.733 | 0.742 | −0.022 | 0.774 | 53.0 % | 안정 |
| **A1 + E1 (한국산림 pos + 산지전용 neg)** | 0.83 | 0.698 | **0.978** | **0.815** | **+0.051** | 0.819 | 16.6 % | 안정 |
| A1 + E3 (한국산림 pos + v1 넓은 neg) | 0.75 | 0.639 | 0.867 | 0.736 | −0.028 | 0.786 | 12.6 % | **불안정 (Δτ 0.22)** |
| A1 + D2 | 0.99 | 0.744 | 0.644 | 0.691 | −0.074 | 0.734 | 67.1 % | 안정 |

- **A1 + E1 = 전체 최고 (test F1 0.815, recall 0.978).** 한국 산림특성 pos ("소나무+갈색 참나무 혼효림,
  조림지 줄심기") + 구체적 산지전용 neg (태양광·건물·도로·벌채나지·온실·묘지·채석) 를 **필지 내부
  패치에만** 적용하면 임상 텍스처 vs 전환 텍스처가 선명하게 갈린다. 전환 필지를 거의 다 잡음(R 0.98).
- **일반 pos(v1)·넓은 neg(E3) 로는 안 됨.** A1 + v1 은 F1 0.71 (최저권), A1 + E3 는 불안정.
  A1 에서는 **pos·neg 둘 다 구체적일 때만** 효과 — 정보량이 적은 내부 패치라 프롬프트가 날카로워야 함.
- 정밀도는 0.70 (마스킹 잔여 경계 패치가 임상 아닌 것을 임상으로) — 고재현 스크리닝용.

## Set FILL — 폴리곤 밖을 검정 대신 다른 값으로

`classify_wholeimage.py --fill {black,mean,gray,inpaint}`. v1 프롬프트 고정, mask 칩, 채움만 교체.

| fill | dev τ\* | test P | test R | **test F1** | ΔvsA | 자체최적 F1 | 밴드(428) | 안정성 |
|---|---|---|---|---|---|---|---|---|
| **black** (현행 = A) | 0.95 | 0.773 | 0.756 | 0.764 | — | 0.775 | 51.2 % | 안정 |
| **mean** (폴리곤 내부 평균색) | 0.94 | **0.850** | 0.756 | **0.800** | **+0.036** | 0.815 | 45.8 % | 안정 |
| gray (ImageNet 상수 회색) | 0.97 | 0.685 | 0.822 | 0.748 | −0.016 | 0.752 | 47.0 % | 안정 |
| inpaint (cv2 TELEA 텍스처 연장) | 0.93 | 0.842 | 0.711 | 0.771 | +0.007 | **0.825** | 31.8 % | 안정 |
| mean + C | 0.93 | 0.814 | 0.778 | 0.795 | +0.031 | 0.825 | 37.6 % | 안정 |
| inpaint + C | 0.86 | 0.865 | 0.711 | 0.780 | +0.016 | 0.825 | **10.5 %** | 안정 |

- **내부 평균색 채움이 검정보다 확실히 낫다: F1 +0.036, 정밀도 0.77→0.85 (dev·test 거의 동일 → 신뢰).**
  검정은 (1) "어두운 장면" 편향, (2) 얇은 필지를 고대비 선 = `a paved road` 로 오인 (v1 최다 attractor
  99). 내부 평균색은 프레임 85 %를 필지 자기 색으로 → 임상 필지는 짙은 녹색 프레임 → "대체로 숲",
  전환 필지는 밝은 갈색 프레임 → "대체로 나지". `a paved road` attractor 99→47.
- **상수 회색은 검정보다 나쁘다 (F1 0.748, P 0.69).** 균일 회색 = CLIP 에게 "평탄 나지·콘크리트" 로
  읽혀 `bare earth cleared of trees`·`a grassy field` 가 최다 attractor → 임상 필지를 nagative 로 끌어당김.
  **채움은 필지 자기 색을 담을 때만 이득**, 범용 상수는 역효과.
- inpaint(텍스처 연장)은 분리도가 가장 좋다 (밴드 10~32 %) 지만 p_forest 분포가 이동해 τ 재튜닝 필요
  (자동 스윕이 처리).

## 종합 (22 config, fig9f Pareto 프런티어)

Pareto 프런티어(P·R 둘 다 우월) = **A1+E1 → context+E3 → mean+C → mean → inpaint+C**.
프런티어에 오른 건 전부 **A1 / context / 채움 교체**. mask whole-image + 프롬프트 교체 점은
**하나도 프런티어에 없음** — baseline A 조차 mean·mean+C 에 지배당함.

| 목적 | 조합 | test P/R/F1 | τ | 특징 |
|---|---|---|---|---|
| **고재현 스크리닝** | A1(필지만) + 한국산림 pos + 산지전용 neg (`set_e1`) | 0.70 / **0.98** / **0.815** | 0.83 | FN 1건. "임야로 통과=거의 확실히 임야" |
| **정밀 판정** | mask + **mean-fill** + v1 | **0.85** / 0.76 / 0.800 | 0.93 | 한 줄 변경. FP 최소 |
| 균형 | mask + mean-fill + C (`set_c`) | 0.81 / 0.78 / 0.795 | 0.93 | P·R 대칭 |
| τ-무관 | context + 한국산림 pos + v1 neg (`set_e3`) | 0.75 / 0.84 / 0.792 | 0.5~0.9 아무값 | 밴드 5 % |

## 채택 판정

A-3 기준 = 베이스라인 대비 test F1 **+0.02 이상**.

- **mask 칩 whole-image + 프롬프트만 (A~D2, 7세트)**: 어느 것도 통과 못함.
- **검정→내부평균색 채움 (Set FILL)**: mask + v1 그대로 **F1 0.764→0.800 (+0.036), 정밀도 0.85**, 안정.
  한 줄 변경. inpaint 채움은 자체최적 0.825.
- **A1 (필지 패치만) + 한국산림 pos + 산지전용 neg**: **F1 0.815 (+0.051), recall 0.978**, 안정.
- **context + 한국산림 pos (E3)**: F1 0.792 (+0.028), τ 0.5~0.94 전구간 0.75~0.79 (τ 비민감).

→ **정밀 판정 프로파일**: mask + **mean-fill** + v1 (F1 0.80, P 0.85), τ≈0.93.
→ **고재현 스크리닝 프로파일**: **A1 + 한국산림 pos + 산지전용 neg** (`03_pipeline/prompts/imya_korean_forest.json`(구 set_e1), F1 0.82,
  R 0.98), τ≈0.83. 또는 context + E3.
→ 프롬프트 문구 자체는 mask whole-image 에서 레버가 아니었음. **레버는 (1) 폴리곤 밖 채움,
  (2) 인코딩(A1/context), (3) 한국 산림특성을 반영한 pos** — 이 셋은 프롬프트 "세트 교체"가 아니라
  파이프라인 구조 변경.

### 심사 피드백 (a) — mask 칩에서 어떤 설계가 F1을 최대화하는가? 서술형이 도움이 되는가?

**서술형은 도움이 안 된다.** mask 칩은 필지 표면만 보이고 밖은 검정 —
장면·주변·재질 디테일을 늘린 서술문(v2·C)은 오히려 검정 편향과 얇은 필지의 "선/도로" 신호와 경합해
정밀도를 떨어뜨린다 (C test P 0.68 vs A 0.77). 네 세트의 **달성 가능 F1(자체최적)은 0.76~0.79로
분할 노이즈(±0.02) 안에서 동률**이고, 지배 레버는 여전히 τ 다.
단, 서술형의 부수 효과가 둘 있다:
(1) v2 는 판정불가밴드를 51 %→19 % 로 좁힌다 (음성 프롬프트가 "숲 맥락에서의 전환"에 특화돼
   순수 임상과 멀어짐) — 운영상 "회색지대" 축소에는 유리.
(2) C 의 `"a wide paved road or graded dirt track cutting across the hillside"` 는 v1 의 최다 오탐원인
   `"a paved road"` 를 정리한다 (형질변경의심 중 관통도로 attractor 34→2, 관련 dev FP 소멸).
   → v1 을 유지하되 이 한 문구만 교체하는 "A+road" 를 후속 후보로 권장.

### 심사 피드백 (b) — 프롬프트 생성 방식 & 오류율

**수작업 설계.** 근거 = 산림청 불법산지전용 유형(벌채나지·정지·관통도로·태양광·시설·경작·묘지·주차야적)
+ 예선 오분류(baseline v1 의 `closest_neg` 분포, τ=0.95). LLM 자동생성은 쓰지 않음.
Set C 의 음성 12문구 + 양성 6문구 중 **dev 75 에서 역효과(임야를 강하게 끌어당김)로 판명된 문구 2개**:
`"a mowed grassy clearing with no trees"` (dev FP 17건 중 11건이 이 문구로 몰림),
`"rows of white plastic greenhouse tunnels"` (dev FP 5건, 형질변경의심 attractor 76건).
→ 초안 오류율 **2 / 18 ≈ 11 %**. C2 에서 전자는 제거(양성 `"scattered trees over rough grass on a
steep hillside"` 로 경계흡수), 후자는 `"long white plastic greenhouse roofs"` 로 완화.
교정 후에도 test F1 이 A 를 넘지 못함 → mask 칩에서 음성 프롬프트 정교화의 한계.

### 심사 피드백 (c) — 지목 모호 · 이용현황 혼재 필지의 오분류

`mixed_ambiguous_subset.csv` = **네 세트가 전부 틀린 필지 29개** (임야 22 = 지속 FP, 형질변경 7 = 지속 FN).

- **지속 FP 22 (실제 임야인데 항상 의심)**: `closest_neg` 이 `a grassy field`(10) · `bare earth cleared
  of trees`(7) · `a cultivated crop field`(4). 육안상 **성긴 유령·유목 임상, 초본층이 드러난 임상, 임연부**.
  valid_ratio 0.15~0.64 로 마스킹 잔여 탓만은 아님. v2(B) 는 이 중 다수를 구제한다
  (산34임 0.94→0.46, 산33임 0.94→0.56) — 음성문구가 "숲에서 벌채된" 식으로 맥락 고정돼 온전한
  성긴 임상엔 안 걸리기 때문. 그래서 B 의 판정불가밴드가 제일 낮다. 그러나 정밀도 총량은 못 살림.
- **지속 FN 7 (실제 형질변경인데 항상 놓침)**: 6/7 이 `closest_neg` = 벌채나지/묘지인데 `p_forest ≈ 0.99`.
  **재식생된 오래된 개간지 · 묘지군** — 단일 시점 겨울 RGB 로 임상과 대비가 거의 없음.
  어떤 프롬프트로도 안 올라옴 (문서의 재현율 천장 ~0.82 와 일치).

**결론**: 혼재·모호 필지는 프롬프트만으로 못 푼다. B 스타일 음성문구가 FP 쪽을 완화하나 총 F1 이득은 없음.
→ 다시점 변화탐지 · 경사도/DEM · 허가·현장 데이터 결합이 필요.

## 발표 콜아웃

### 3-3 🔴 (프롬프트 방식·오류율)

> 프롬프트는 **수작업 설계**(산림청 전용유형 + 예선 오분류 `closest_neg` 분포). LLM 자동생성 미사용.
> mask 특화 신규 세트(C)의 초안 18문구 중 dev 검증에서 **2문구(≈11 %)가 임야를 과흡인**해 폐기·교정.
> 교정 후에도 test F1 은 baseline(v1, 0.764)을 못 넘음.

### 4-4 🔴 (mask 칩 프롬프트 설계 결론)

> mask 칩에서는 **서술형 프롬프트가 이득이 없다.** v1/v2/C/C2 의 달성 F1 은 0.76~0.79 로
> dev/test 분할 노이즈(±0.02) 안 동률이고, τ 가 지배 레버.
> 서술형의 유일한 실익은 **판정불가밴드 축소**(v2: 51 %→19 %)와 관통도로 오탐원 정리.
> → 확정: **v1 유지** (`prompts/final_imya.json`), τ\*≈0.95.
> 후속 후보: v1 + 관통도로 문구 1개만 교체(A+road), 그리고 다시점/DEM 결합.

## 파일

```
02_experiments/prompt_ablation/prompt_sets/*.json
  set_e1.json  set_e2.json  set_e3.json  final_imya.json
02_experiments/results/prompt_ablation/
  <set>/scores.csv       428필지 p_forest (classify_wholeimage.py)
  <set>/metrics.json     dev/test P·R·F1·혼동행렬·τ*·판정불가밴드·안정성 (run_ablation.py)
  <set>/errors.csv       dev+test FP·FN 목록
  <set>/closest_neg.csv  형질변경의심(428) closest_neg 분포
  summary.csv            세트 비교표
  fig9_prompt_ablation.png   [그림 9] mask 세트
  fig9c_context.png          [그림 9c] context 세트 (E0~E3)
  mixed_ambiguous_subset.csv 혼재·모호 서브셋 (네 세트 공통 오분류 29)
```

재현:
```
python 01_preprocess/scripts/labeling/make_split.py                      # split.csv (1회, 커밋 후 재생성 금지)
for s in set_a set_b set_c set_c2 set_c_nowinter set_d set_d2; do  # mask: --chips 01_preprocess/chips/37714092_mask
  python 02_experiments/image_method/classify_wholeimage.py --cpu --chips 01_preprocess/chips/37714092_mask \
    --prompts-file 02_experiments/prompt_ablation/prompt_sets/$s.json --out 02_experiments/results/prompt_ablation/$s/scores.csv --no-montage
  python 02_experiments/prompt_ablation/run_ablation.py \
    --scores 02_experiments/results/prompt_ablation/$s/scores.csv \
    --labels 01_preprocess/labels/imya_eval_150/labels.csv \
    --split 01_preprocess/labels/imya_eval_150/split.csv --delta 0.05 \
    --out 02_experiments/results/prompt_ablation/$s/metrics.json
done
python 02_experiments/prompt_ablation/report.py              # summary.csv + fig9
```
