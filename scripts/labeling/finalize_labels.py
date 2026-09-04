#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
label_tool.html 에서 내보낸 labels.json 을 받아 정리 + (있으면) 방법별 성능 채점.

- data/labels/imya_eval_150/labels.csv        (pnu,jibun,area_m2,label)
- data/labels/imya_eval_150/임야|형질변경|보류/  미리보기(ctx) 복사
- 방법 CSV(data/remoteclip/*.csv)가 있으면 형질변경=positive 로 P/R/F1 출력 (보류 제외)

사용:
    python scripts/labeling/finalize_labels.py C:/Users/.../Downloads/labels.json
    python scripts/labeling/finalize_labels.py            # out 폴더나 ~/Downloads 에서 자동 탐색
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

OUT = Path("data/labels/imya_eval_150")
PREV = OUT / "previews"
POS = "형질변경"

# (표시이름, csv경로, 예측라벨컬럼, positive 값)
METHODS = [
    ("v1 CLS",        "data/remoteclip/37714092_imya_binary.csv",     "label", "형질변경의심"),
    ("A1 토큰드롭",   "data/remoteclip/37714092_imya_binary_a1.csv",  "label", "형질변경의심"),
    ("C2 dense",      "data/remoteclip/37714092_imya_c2.csv",         "label", "형질변경의심"),
]


def find_json(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    for c in [OUT / "labels.json", Path.home() / "Downloads" / "labels.json", Path("labels.json")]:
        if c.exists():
            return c
    raise SystemExit("labels.json 을 못 찾음. 경로를 인자로 주세요.")


def load_pred(path: str, col: str, pos: str) -> dict[str, bool]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8-sig") as f:
        return {r.get("pnu", ""): (r.get(col) == pos) for r in csv.DictReader(f)}


def score(gt: dict[str, str], pred: dict[str, bool]) -> str:
    tp = fp = fn = tn = 0
    n = 0
    for pnu, g in gt.items():
        if g == "보류" or pnu not in pred:
            continue
        n += 1
        gp = (g == POS)
        pp = pred[pnu]
        tp += gp and pp
        fp += (not gp) and pp
        fn += gp and (not pp)
        tn += (not gp) and (not pp)
    if n == 0:
        return "  (매칭 0)"
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / n
    return (f"  n={n:<3} P={prec:.2f} R={rec:.2f} F1={f1:.2f} Acc={acc:.2f}  "
            f"(TP{tp} FP{fp} FN{fn} TN{tn})")


def main() -> None:
    jp = find_json(sys.argv[1] if len(sys.argv) > 1 else None)
    arr = json.loads(jp.read_text(encoding="utf-8"))
    gt = {r["pnu"]: r["label"] for r in arr}
    print(f"labels.json: {jp}  ({len(gt)}개)")

    rows = sorted(arr, key=lambda r: r["pnu"])
    with open(OUT / "labels.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["pnu", "jibun", "area_m2", "label", "ts"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(f"→ {OUT / 'labels.csv'}")

    from collections import Counter
    cc = Counter(gt.values())
    print("분포: " + " · ".join(f"{k} {v}" for k, v in cc.most_common()))

    for sub in ["임야", "형질변경", "보류"]:
        d = OUT / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    miss = 0
    for pnu, lab in gt.items():
        src = PREV / f"{pnu}_ctx.jpg"
        if src.exists():
            shutil.copy2(src, OUT / lab / f"{pnu}.jpg")
        else:
            miss += 1
    print(f"미리보기 분류 복사 완료 (누락 {miss})")

    print("\n── 방법별 성능 (형질변경 = positive, 보류 제외) ──")
    for name, path, col, pos in METHODS:
        pred = load_pred(path, col, pos)
        print(f"{name:<12}{score(gt, pred) if pred else '  (CSV 없음: ' + path + ')'}")


if __name__ == "__main__":
    main()
