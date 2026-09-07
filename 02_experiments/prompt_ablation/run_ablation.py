#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
한 실험 산출 CSV 를 손라벨 + dev/test 분할로 채점.

프로토콜:
  - dev 에서 임계 스윕 → F1 최대 평탄대 중앙 τ*  (common.scoring.best_tau)
  - test 는 그 τ* 로 1회 채점
  - positive = 형질변경
  - `--predict below` : score < τ  → positive  (p_forest 계열, 기본)
    `--predict above` : score >= τ → positive  (built_frac 등)
  - 판정불가 밴드: |score − τ*| ≤ δ. 전체 / test 각각 비율
  - 안정성: dev τ* 와 test 자체최적 τ 의 차 > 0.10 이면 "불안정"

출력:
  <out>                       metrics.json
  <out디렉토리>/errors.csv    dev+test 의 FP·FN
  <out디렉토리>/closest_neg.csv  형질변경의심(전체) 의 closest_neg 분포

사용:
  python 02_experiments/prompt_ablation/run_ablation.py \
    --scores 02_experiments/results/.../scores.csv --out .../metrics.json
  # densepatch 계열:
  python 02_experiments/prompt_ablation/run_ablation.py --scores ... --out ... \
    --score-col built_frac --predict above --tau-lo 0.0 --tau-hi 0.6
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import LABEL_SET  # noqa: E402
from common.scoring import read_csv, confusion, best_tau, band_frac  # noqa: E402

POS_LABEL = "형질변경"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=LABEL_SET / "labels.csv")
    ap.add_argument("--split", type=Path, default=LABEL_SET / "split.csv")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--score-col", default="p_forest", help="채점에 쓸 점수 컬럼 (기본 p_forest)")
    ap.add_argument("--predict", choices=["below", "above"], default="below",
                    help="below: score < τ → positive / above: score >= τ → positive")
    ap.add_argument("--tau-lo", type=float, default=0.50)
    ap.add_argument("--tau-hi", type=float, default=0.99)
    ap.add_argument("--tau-step", type=float, default=0.01)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    col, predict = args.score_col, args.predict
    kw = dict(lo=args.tau_lo, hi=args.tau_hi, step=args.tau_step)

    scores = {r["pnu"]: r for r in read_csv(args.scores)}
    for r in scores.values():
        r["_s"] = float(r[col])
    labels = {r["pnu"]: r["label"] for r in read_csv(args.labels)}
    split = {r["pnu"]: r["split"] for r in read_csv(args.split)}
    all_s = [r["_s"] for r in scores.values()]

    subsets: dict[str, list[dict]] = {"dev": [], "test": []}
    miss = 0
    for pnu, sp in split.items():
        g = labels.get(pnu)
        if g == "보류" or g is None:
            continue
        s = scores.get(pnu)
        if s is None:
            miss += 1
            continue
        subsets[sp].append({"pnu": pnu, "jibun": s.get("jibun", ""), "s": s["_s"], "gt": g,
                            "closest_neg": s.get("closest_neg", "")})

    def pr(rows):
        return [(r["s"], r["gt"] == POS_LABEL) for r in rows]

    dev_best = best_tau(pr(subsets["dev"]), predict=predict, **kw)
    tau_star = dev_best["tau"]
    test_at = confusion(pr(subsets["test"]), tau_star, predict=predict)
    test_best = best_tau(pr(subsets["test"]), predict=predict, **kw)
    dev_at_star = confusion(pr(subsets["dev"]), tau_star, predict=predict)
    unstable = abs(tau_star - test_best["tau"]) > 0.10

    b_all = band_frac(all_s, tau_star, args.delta)
    b_test = band_frac([r["s"] for r in subsets["test"]], tau_star, args.delta)

    def is_pos(v):
        return v < tau_star if predict == "below" else v >= tau_star

    susp_neg = Counter(r.get("closest_neg", "") for r in scores.values()
                       if is_pos(r["_s"]) and r.get("closest_neg"))

    metrics = {
        "scores": str(args.scores), "score_col": col, "predict": predict,
        "n_dev": len(subsets["dev"]), "n_test": len(subsets["test"]), "n_missing": miss,
        "tau_star": tau_star,
        "dev": {**dev_at_star, "sweep_best": dev_best},
        "test": {**test_at, "sweep_best_tau": test_best["tau"],
                 "sweep_F1_at_that_tau": test_best["F1"], "sweep_F1_max": test_best["f1_max"]},
        "stability": {"dev_tau_star": tau_star, "test_best_tau": test_best["tau"],
                      "delta_tau": round(abs(tau_star - test_best["tau"]), 2),
                      "unstable": unstable},
        "undecidable_band": {
            "delta": args.delta,
            "all": {"n": b_all[0], "total": b_all[1], "frac": b_all[2]},
            "test": {"n": b_test[0], "total": b_test[1], "frac": b_test[2]},
        },
        "closest_neg_suspect": susp_neg.most_common(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    err_path = args.out.parent / "errors.csv"
    with open(err_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["split", "pnu", "jibun", col, "gt", "pred", "closest_neg", "err"])
        for sp in ("dev", "test"):
            for r in sorted(subsets[sp], key=lambda r: r["s"]):
                gt_pos = r["gt"] == POS_LABEL
                pred_pos = is_pos(r["s"])
                if gt_pos == pred_pos:
                    continue
                w.writerow([sp, r["pnu"], r["jibun"], round(r["s"], 4), r["gt"],
                            "형질변경의심" if pred_pos else "임야", r["closest_neg"],
                            "FP" if pred_pos else "FN"])

    neg_path = args.out.parent / "closest_neg.csv"
    with open(neg_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["closest_neg", "count"])
        for k, v in susp_neg.most_common():
            w.writerow([k, v])

    d, t = dev_at_star, test_at
    print(f"scores={args.scores}  col={col} predict={predict}")
    print(f"  n_dev={len(subsets['dev'])}  n_test={len(subsets['test'])}  누락={miss}")
    print(f"  τ* (dev F1 최대) = {tau_star}")
    print(f"  dev  @τ*  P={d['P']:.3f} R={d['R']:.3f} F1={d['F1']:.3f}  "
          f"(TP{d['TP']} FP{d['FP']} FN{d['FN']} TN{d['TN']})")
    print(f"  test @τ*  P={t['P']:.3f} R={t['R']:.3f} F1={t['F1']:.3f}  "
          f"(TP{t['TP']} FP{t['FP']} FN{t['FN']} TN{t['TN']})")
    print(f"  test 자체최적 τ={test_best['tau']} F1={test_best['F1']:.3f}  "
          f"→ {'불안정' if unstable else '안정'} (Δτ={abs(tau_star - test_best['tau']):.2f})")
    print(f"  판정불가밴드(±{args.delta}): 전체 {b_all[0]}/{b_all[1]} ({b_all[2]:.1%})  "
          f"test {b_test[0]}/{b_test[1]} ({b_test[2]:.1%})")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
