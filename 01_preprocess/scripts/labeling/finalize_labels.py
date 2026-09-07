#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
label_tool.html 에서 내보낸 labels.json 을 정리 + (있으면) 방법별 성능 채점.

산출 (01_preprocess/labels/imya_eval_150/ 안):
  - labels.csv                     (pnu,jibun,area_m2,label,ts)
  - 임야|형질변경|보류/            미리보기(ctx) 분류 복사

채점: `--pred <scores.csv> ...` 로 준 CSV(들) 을 형질변경=positive 로 P/R/F1/Acc 출력 (보류 제외).
      각 CSV 는 pnu + label 컬럼(값 '형질변경의심') 을 가져야 한다.
      아무것도 안 주면 03_pipeline/runs/*/3_labels.csv 를 자동 수집.

사용:
    python 01_preprocess/scripts/labeling/finalize_labels.py <labels.json 경로>
    python 01_preprocess/scripts/labeling/finalize_labels.py            # labels.json 자동 탐색
    python 01_preprocess/scripts/labeling/finalize_labels.py --pred 03_pipeline/runs/imya_precision/3_labels.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.paths import LABEL_SET, PIPELINE_RUNS  # noqa: E402
from common.scoring import score_labels  # noqa: E402

PREV = LABEL_SET / "previews"
POS = "형질변경"
PRED_POS_VALUE = "형질변경의심"


def find_json(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    for c in [LABEL_SET / "labels.json", Path.home() / "Downloads" / "labels.json", Path("labels.json")]:
        if c.exists():
            return c
    raise SystemExit("labels.json 을 못 찾음. 경로를 인자로 주세요.")


def load_pred(path: Path, col: str = "label", pos: str = PRED_POS_VALUE) -> dict[str, bool]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig") as f:
        return {r.get("pnu", ""): (r.get(col) == pos) for r in csv.DictReader(f)}


def fmt(m: dict) -> str:
    return (f"  n={m['TP'] + m['FP'] + m['FN'] + m['TN']:<3} "
            f"P={m['P']:.2f} R={m['R']:.2f} F1={m['F1']:.2f} Acc={m['Acc']:.2f}  "
            f"(TP{m['TP']} FP{m['FP']} FN{m['FN']} TN{m['TN']})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels_json", nargs="?", default=None)
    ap.add_argument("--pred", type=Path, nargs="*", default=None,
                    help="채점할 예측 CSV(들). 미지정 시 03_pipeline/runs/*/3_labels.csv 자동")
    args = ap.parse_args()

    jp = find_json(args.labels_json)
    arr = json.loads(jp.read_text(encoding="utf-8"))
    gt = {r["pnu"]: r["label"] for r in arr}
    print(f"labels.json: {jp}  ({len(gt)}개)")

    rows = sorted(arr, key=lambda r: r["pnu"])
    with open(LABEL_SET / "labels.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["pnu", "jibun", "area_m2", "label", "ts"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(f"→ {LABEL_SET / 'labels.csv'}")
    print("분포: " + " · ".join(f"{k} {v}" for k, v in Counter(gt.values()).most_common()))

    for sub in ("임야", "형질변경", "보류"):
        d = LABEL_SET / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    miss = 0
    for pnu, lab in gt.items():
        src = PREV / f"{pnu}_ctx.jpg"
        if src.exists():
            shutil.copy2(src, LABEL_SET / lab / f"{pnu}.jpg")
        else:
            miss += 1
    print(f"미리보기 분류 복사 완료 (누락 {miss})")

    preds = args.pred if args.pred is not None else sorted(PIPELINE_RUNS.glob("*/3_labels.csv"))
    if preds:
        print("\n── 방법별 성능 (형질변경 = positive, 보류 제외) ──")
        for p in preds:
            p = Path(p)
            name = p.parent.name if p.name == "3_labels.csv" else p.stem
            pred = load_pred(p)
            print(f"{name:<24}{fmt(score_labels(gt, pred, POS)) if pred else '  (CSV 없음/빈값: ' + str(p) + ')'}")
    else:
        print("\n(채점할 예측 CSV 없음 — --pred 로 지정하거나 파이프라인을 먼저 실행)")


if __name__ == "__main__":
    main()
