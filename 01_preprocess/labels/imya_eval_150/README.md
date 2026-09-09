# 임야 손라벨 세트 (150)

임야 428필지 중 면적 5분위 층화표집 150개. zero-shot 방법(v1/A1/C2 등) 평가용.

## 라벨링 방법

1. `label_tool.html` 을 브라우저로 연다 (더블클릭)
2. 각 필지: 빨간 영역이 판정 대상
   - 나무 수관이 덮여있으면 **임야** (키 `1`)
   - 벌채·나지·조성·건물·주차/야적·태양광·묘지·관통도로 등이면 **형질변경** (키 `2`)
   - 판단 어려우면 **보류** (키 `3`)
   - 칩/영상 자체가 이상(구름·엉뚱한 필지·심한 왜곡)이면 **제외** (키 `4`) — 평가 라벨로 쓰지 않음
   - `←` `→` 로 이동, 라벨은 브라우저에 자동저장
3. 다 하면 상단 **결과 저장 (labels.json)** → 다운로드
4. `python scripts/labeling/finalize_labels.py <labels.json 경로>`

## 파일

- `label_tool.html`   자체완결 라벨링 도구 (이미지 내장, 13.6MB)
- `previews/`          필지별 미리보기 `<PNU>_ctx.jpg`(맥락) `<PNU>_tight.jpg`(근접)
- `manifest.csv`       idx, pnu, jibun, area_m2
- `labels.csv`         (finalize 후) pnu, jibun, area_m2, label — `제외` 는 빠짐
- `excluded.csv`       (finalize 후) 손라벨 `제외` 목록 (칩/영상 이상, 평가 미사용)
- `임야/ 형질변경/ 보류/ 제외/`  (finalize 후) 미리보기 분류 복사

## finalize 출력

`labels.csv` (+ `excluded.csv`) + 폴더 분류 + 방법별 성능(형질변경=positive, 보류·제외 제외) P/R/F1/Acc.
