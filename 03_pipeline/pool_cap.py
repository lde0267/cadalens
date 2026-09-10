#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""검증 지역들을 하나로 합친 recall@budget(CAP) 곡선.

각 runs/<region>/2_scores.csv 의 p_screen(=위반 의심도) 을 위험 점수로 쓰고,
data/<region>/labels.json 의 손라벨(형질변경 = positive)로 채점한다.
지역별 곡선을 합쳐 하나의 pooled 곡선 + 지역별 rec@20% 표를 출력.

    python 03_pipeline/pool_cap.py                       # 기본 6개 검증 도엽(세종 파일럿 36710022 제외)
    python 03_pipeline/pool_cap.py --group imya          # 임야만
    python 03_pipeline/pool_cap.py --only 356 367        # sheet 필터
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE.parent))
DATA = PIPE / "data"
RUNS = PIPE / "runs"
POS = "형질변경"
PILOT_SHEET = "36710022"  # 세종 파일럿 → 검증셋에서 제외


def load_region(region: str):
    sc = RUNS / region / "2_scores.csv"
    lb = DATA / region / "labels.json"
    if not (sc.exists() and lb.exists()):
        return None
    scores = {}
    with open(sc, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            scores[r["pnu"]] = float(r["p_screen"] or 0)
    labels = json.loads(lb.read_text(encoding="utf-8"))["labels"]
    pairs = []
    for r in labels:
        g = r["label"]
        if g in ("보류", "제외"):
            continue
        s = scores.get(r["pnu"])
        if s is None:
            continue
        pairs.append((s, 1 if g == POS else 0))
    return pairs


def cap(pairs):
    s = np.array([p for p, _ in pairs], float)
    y = np.array([p for _, p in pairs], int)
    o = np.argsort(-s, kind="mergesort")
    yy = y[o]
    n = len(yy)
    npos = int(y.sum())
    cum = np.cumsum(yy) / npos
    frac = (np.arange(n) + 1) / n

    def rec_at(f):
        return float(cum[max(1, int(round(f * n))) - 1])

    from common.scoring import roc_auc, ap
    return {
        "n": n, "npos": npos, "prev": npos / n,
        "frac": np.r_[0, frac], "cum": np.r_[0, cum],
        "auroc": roc_auc(s, y, "above"), "ap": ap(s, y, "above"),
        "rec": {f: rec_at(f) for f in (0.05, 0.10, 0.20, 0.30, 0.50)},
    }


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--group", choices=["imya", "nongji"])
    ap_.add_argument("--only", nargs="*", help="sheet 부분일치 필터")
    ap_.add_argument("--keep-pilot", action="store_true", help="세종 파일럿 도엽도 포함")
    ap_.add_argument("--out", default=str(RUNS / "_pool_cap.png"))
    ap_.add_argument("--title", default="검증 지역 통합 · recall@budget")
    args = ap_.parse_args()

    regs = []
    for p in sorted(DATA.iterdir()):
        if not (p.is_dir() and (p / "labels.json").exists()):
            continue
        name = p.name
        grp = "nongji" if name.endswith("_nongji") else "imya"
        if args.group and grp != args.group:
            continue
        if not args.keep_pilot and name.startswith(PILOT_SHEET + "_"):
            continue
        if args.only and not any(o in name for o in args.only):
            continue
        regs.append(name)

    print(f"지역 {len(regs)}개: {', '.join(regs)}\n")
    pooled, per = [], []
    for name in regs:
        pr = load_region(name)
        if not pr or sum(y for _, y in pr) == 0:
            print(f"  {name}: skip")
            continue
        c = cap(pr)
        per.append((name, c))
        pooled += pr

    P = cap(pooled)
    print(f"{'region':22s} {'n':>5} {'pos':>4} {'prev':>5} {'AUROC':>6} {'r@10':>5} {'r@20':>5} {'r@30':>5}")
    for name, c in per:
        print(f"{name:22s} {c['n']:>5} {c['npos']:>4} {c['prev']:>5.2f} {c['auroc']:>6.3f} "
              f"{c['rec'][0.10]:>5.2f} {c['rec'][0.20]:>5.2f} {c['rec'][0.30]:>5.2f}")
    print("-" * 70)
    print(f"{'POOLED':22s} {P['n']:>5} {P['npos']:>4} {P['prev']:>5.2f} {P['auroc']:>6.3f} "
          f"{P['rec'][0.10]:>5.2f} {P['rec'][0.20]:>5.2f} {P['rec'][0.30]:>5.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for name, c in per:
        ax.plot(c["frac"], c["cum"], lw=0.8, alpha=0.35, color="gray")
    ax.plot(P["frac"], P["cum"], lw=2.6, color="#1f77b4",
            label=f"통합 모델 (AUC {P['auroc']:.2f}, n={P['n']:,})")
    ax.plot([0, 1], [0, 1], "--", c="gray", lw=1, label="무작위 조사")
    ax.plot([0, P["prev"], 1], [0, 1, 1], ":", c="green", lw=1, label="이상적")
    ax.axvline(0.20, c="orange", lw=1, ls="-")
    ax.annotate(f"예산 20% → 위반 {P['rec'][0.20]*100:.0f}% 검출",
                xy=(0.20, P["rec"][0.20]), xytext=(0.28, max(0.12, P["rec"][0.20] - 0.22)),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="orange"))
    ax.set_xlabel("조사한 필지 비율 (예산)")
    ax.set_ylabel("찾아낸 위반 비율 (recall)")
    ax.set_title(args.title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
