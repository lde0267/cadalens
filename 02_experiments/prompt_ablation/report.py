#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
프롬프트 세트별 metrics.json 을 모아 summary.csv + [그림 9] PNG 생성.

[그림 9]
  (좌) 세트별 test P / R / F1 그룹 막대
  (우) 대표 필지 5개(관통도로·태양광·묘지·성긴임상·울창임상) 세트별 p_forest 표

사용:
  python scripts/imya/prompt_ablation_report.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import EXPERIMENT_RESULTS, PROMPT_SETS_DIR  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = EXPERIMENT_RESULTS / "prompt_ablation"
SETS = [
    ("A · v1 (일반명사, baseline)", "set_a"),
    ("B · v2 (서술형, real-fit용)", "set_b"),
    ("C · mask특화 (서술형 신규)", "set_c"),
    ("D · 산지전용 빈발유형 (도로 포함)", "set_d"),
    ("D2 · D 자연어꼬리 제거", "set_d2"),
]
# 진단 전용(그림9 제외, summary 에는 포함)
DIAG = [
    ("C2 · C -초지 +온실완화", "set_c2"),
    ("C-nowinter · C2 pos에서 겨울 제거", "set_c_nowinter"),
]
# context 칩 실험 (fig9c)
CTX = [
    ("E0 · context + v1", "set_e0"),
    ("E1 · context + 한국산림pos + 산지전용neg", "set_e1"),
    ("E2 · context + 일반pos + 산지전용neg", "set_e2"),
    ("E3 · context + 한국산림pos + v1 넓은neg", "set_e3"),
]
# A1 토큰드롭(필지만) 실험 (fig9d)
A1 = [
    ("A1 + v1", "set_a1_v1"),
    ("A1 + C (mask서술)", "set_a1_c"),
    ("A1 + E1 (한국산림pos+산지전용neg)", "set_a1_e1"),
    ("A1 + E3 (한국산림pos+v1neg)", "set_a1_e3"),
    ("A1 + D2", "set_a1_d2"),
]
# mask 폴리곤 밖 채움 방식 실험, v1 프롬프트 고정 (fig9e)
FILL = [
    ("black (현행=A)", "set_a"),
    ("mean (내부 평균색)", "set_fill_mean"),
    ("gray (상수 회색)", "set_fill_gray"),
    ("inpaint (텍스처 연장)", "set_fill_inpaint"),
    ("mean + C", "set_fill_mean_c"),
    ("inpaint + C", "set_fill_inpaint_c"),
]
# 대표 필지: (유형, pnu, jibun, gt)
REPR = [
    ("관통도로", "4155012600200480012", "산48-12임", "형질변경"),
    ("태양광", "4155031034200830000", "산83임", "형질변경"),
    ("묘지", "4155012600200250069", "산25-69임", "형질변경"),
    ("성긴 임상", "4155012600200480010", "산48-10임", "임야"),
    ("울창 임상", "4155012600200180047", "산18-47임", "임야"),
]

for p in (r"C:\Windows\Fonts\malgun.ttf",):
    if Path(p).exists():
        fm.fontManager.addfont(p)
        plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
plt.rcParams["axes.unicode_minus"] = False


def load_metrics(slug: str) -> dict:
    return json.loads((ROOT / slug / "metrics.json").read_text(encoding="utf-8"))


def load_scores(slug: str) -> dict:
    with open(ROOT / slug / "scores.csv", encoding="utf-8-sig") as f:
        return {r["pnu"]: float(r["p_forest"]) for r in csv.DictReader(f)}


def _uniq(pairs):
    seen, out = set(), []
    for n, s in pairs:
        if s not in seen:
            seen.add(s); out.append((n, s))
    return out


def main() -> None:
    all_sets = _uniq(SETS + DIAG + CTX + A1 + FILL)
    mets = {slug: load_metrics(slug) for _, slug in all_sets}
    scrs = {slug: load_scores(slug) for _, slug in SETS + CTX}
    a_f1 = mets["set_a"]["test"]["F1"]

    # ---- summary.csv (플롯 세트 + 진단 세트)
    out_csv = ROOT / "summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["set", "slug", "pos_n/neg_n", "dev_tau_star", "dev_F1",
                    "test_P", "test_R", "test_F1", "dF1_vs_A",
                    "test_selfbest_tau", "test_selfbest_F1",
                    "undecidable_pct_428", "undecidable_pct_test", "stability"])
        for name, slug in all_sets:
            m = mets[slug]
            pj = PROMPT_SETS_DIR / f"{slug}.json"
            pd = json.loads(pj.read_text(encoding="utf-8")) if pj.exists() else {"positive": [], "negative": []}
            w.writerow([
                name, slug, f'{len(pd["positive"])}/{len(pd["negative"])}',
                m["tau_star"], m["dev"]["F1"],
                m["test"]["P"], m["test"]["R"], m["test"]["F1"],
                round(m["test"]["F1"] - a_f1, 4),
                m["test"]["sweep_best_tau"], m["test"]["sweep_F1_max"],
                m["undecidable_band"]["all428"]["frac"],
                m["undecidable_band"]["test"]["frac"],
                "불안정" if m["stability"]["unstable"] else "안정",
            ])
    print(f"→ {out_csv}")

    # ---- figure 9
    fig = plt.figure(figsize=(15, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.18)

    ax = fig.add_subplot(gs[0, 0])
    labels = [n for n, _ in SETS]
    P = [mets[s]["test"]["P"] for _, s in SETS]
    R = [mets[s]["test"]["R"] for _, s in SETS]
    F = [mets[s]["test"]["F1"] for _, s in SETS]
    x = range(len(SETS))
    bw = 0.26
    b1 = ax.bar([i - bw for i in x], P, bw, label="Precision", color="#4C78A8")
    b2 = ax.bar(list(x), R, bw, label="Recall", color="#F58518")
    b3 = ax.bar([i + bw for i in x], F, bw, label="F1", color="#54A24B")
    for bars in (b1, b2, b3):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01,
                    f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(a_f1, ls="--", lw=1, color="#54A24B", alpha=0.7)
    ax.text(len(SETS) - 0.5, a_f1 + 0.005, f"A F1={a_f1:.3f}", fontsize=8,
            color="#54A24B", ha="right")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9, rotation=12, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("test (75필지) 점수")
    ax.set_title("[그림 9-좌] 프롬프트 세트별 test P / R / F1  (형질변경 = positive, dev-fit τ*)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    col_labels = ["대표 필지", "실제 라벨"] + [n.split(" · ")[0] for n, _ in SETS]
    cell = []
    for typ, pnu, jibun, gt in REPR:
        row = [f"{typ}\n{jibun}", gt] + [f"{scrs[s][pnu]:.2f}" for _, s in SETS]
        cell.append(row)
    tbl = ax2.table(cellText=cell, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.12, 3.0)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#EEEEEE")
        tbl[0, j].set_text_props(weight="bold")
    for i, (typ, pnu, jibun, gt) in enumerate(REPR, start=1):
        for j in range(2, len(col_labels)):
            v = scrs[SETS[j - 2][1]][pnu]
            flag = v < mets[SETS[j - 2][1]]["tau_star"]
            correct = (flag and gt == "형질변경") or ((not flag) and gt == "임야")
            tbl[i, j].set_facecolor("#DCEEDC" if correct else "#F6D7D7")
    ax2.set_title("[그림 9-우] 대표 필지 p_forest  (초록=세트 τ* 기준 정답, 빨강=오답)",
                  fontsize=10)

    fig.suptitle("임야 파이프라인 · mask 칩 · 프롬프트 ablation  "
                 "(dev 75 에서 τ* 스윕 → test 75 1회 채점)", fontsize=11)
    out_png = ROOT / "fig9_prompt_ablation.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"→ {out_png}")

    # ---- figure 9c : context 칩 실험
    figc, axc = plt.subplots(figsize=(9.5, 5.2))
    cl = [n for n, _ in CTX]
    cx = range(len(CTX))
    for k, (key, col) in enumerate([("P", "#4C78A8"), ("R", "#F58518"), ("F1", "#54A24B")]):
        vals = [mets[s][ "test"][key] for _, s in CTX]
        bb = axc.bar([i + (k - 1) * 0.26 for i in cx], vals, 0.26,
                     label={"P": "Precision", "R": "Recall", "F1": "F1"}[key], color=col)
        for r in bb:
            axc.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01,
                     f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    axc.axhline(a_f1, ls="--", lw=1, color="#54A24B", alpha=0.7)
    axc.text(len(CTX) - 0.5, a_f1 + 0.006, f"mask baseline A F1={a_f1:.3f}", fontsize=8,
             color="#54A24B", ha="right")
    axc.set_xticks(list(cx))
    axc.set_xticklabels(cl, fontsize=8.5, rotation=12, ha="right")
    axc.set_ylim(0, 1.05)
    axc.set_ylabel("test (75필지) 점수")
    axc.set_title("[그림 9c] context 칩 · 프롬프트 세트별 test P / R / F1  (dev-fit τ*)", fontsize=10)
    axc.legend(fontsize=8, loc="lower right")
    axc.grid(axis="y", alpha=0.25)
    out_png_c = ROOT / "fig9c_context.png"
    figc.savefig(out_png_c, dpi=140, bbox_inches="tight")
    print(f"→ {out_png_c}")

    # ---- figure 9d / 9e : A1(필지만), FILL(폴리곤 밖 채움)
    for grp, fname, title in [
        (A1, "fig9d_a1.png", "[그림 9d] A1 토큰드롭(모델이 필지 패치만) · test P / R / F1  (dev-fit τ*)"),
        (FILL, "fig9e_fill.png", "[그림 9e] mask 폴리곤 밖 채움 방식 · v1 프롬프트 고정 · test P / R / F1"),
    ]:
        figg, axg = plt.subplots(figsize=(10.5, 5.2))
        gx = range(len(grp))
        for k, (key, col) in enumerate([("P", "#4C78A8"), ("R", "#F58518"), ("F1", "#54A24B")]):
            bb = axg.bar([i + (k - 1) * 0.26 for i in gx],
                         [mets[s]["test"][key] for _, s in grp], 0.26,
                         label={"P": "Precision", "R": "Recall", "F1": "F1"}[key], color=col)
            for r in bb:
                axg.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01,
                         f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=8)
        axg.axhline(a_f1, ls="--", lw=1, color="#54A24B", alpha=0.7)
        axg.text(len(grp) - 0.5, a_f1 + 0.006, f"baseline A F1={a_f1:.3f}", fontsize=8,
                 color="#54A24B", ha="right")
        axg.set_xticks(list(gx))
        axg.set_xticklabels([n for n, _ in grp], fontsize=8.5, rotation=12, ha="right")
        axg.set_ylim(0, 1.05)
        axg.set_ylabel("test (75필지) 점수")
        axg.set_title(title, fontsize=10)
        axg.legend(fontsize=8, loc="lower right")
        axg.grid(axis="y", alpha=0.25)
        figg.savefig(ROOT / fname, dpi=140, bbox_inches="tight")
        print(f"→ {ROOT / fname}")

    # ---- figure 9f : 전 config P–R 산점 + Pareto 프런티어
    figf, axf = plt.subplots(figsize=(9.5, 7.5))
    grp_of = {}
    for n, s in SETS + DIAG:
        grp_of[s] = ("mask whole-image (프롬프트)", "#4C78A8", "o")
    for n, s in CTX:
        grp_of[s] = ("context 칩", "#F58518", "s")
    for n, s in A1:
        grp_of[s] = ("A1 (필지 패치만)", "#54A24B", "^")
    for n, s in FILL:
        grp_of[s] = ("mask + 채움 교체", "#B279A2", "D")
    grp_of["set_a"] = ("mask whole-image (프롬프트)", "#4C78A8", "o")
    pts = []
    seen = set()
    for name, slug in all_sets:
        if slug in seen:
            continue
        seen.add(slug)
        m = mets[slug]
        P, R, F1 = m["test"]["P"], m["test"]["R"], m["test"]["F1"]
        g, col, mk = grp_of.get(slug, ("기타", "#888", "x"))
        pts.append((P, R, F1, name, slug, g, col, mk))
        axf.scatter(P, R, s=70, c=col, marker=mk, edgecolors="white", linewidths=0.6, zorder=3)
    # Pareto 프런티어 (P, R 둘 다 클수록 좋음)
    fr = []
    for p in sorted(pts, key=lambda t: (-t[0], -t[1])):
        if all(not (o[0] >= p[0] and o[1] >= p[1] and o != p) for o in pts):
            fr.append(p)
    fr.sort(key=lambda t: t[0])
    axf.plot([p[0] for p in fr], [p[1] for p in fr], "--", color="#333", lw=1, zorder=2,
             label="Pareto 프런티어")
    for p in pts:
        star = p[3] in ("A1 + E1 (한국산림pos+산지전용neg)", "mean (내부 평균색)",
                        "E3 · context + 한국산림pos + v1 넓은neg", "A · v1 (일반명사, baseline)")
        if star or p in fr:
            axf.annotate(f"{p[3].split(' (')[0].split(' · ')[0]}  F1={p[2]:.2f}",
                         (p[0], p[1]), fontsize=7.5, xytext=(4, 4),
                         textcoords="offset points")
    for f1v in (0.70, 0.75, 0.80):
        xs = [x / 100 for x in range(40, 101)]
        ys = [f1v * x / (2 * x - f1v) if 2 * x - f1v > 0 else None for x in xs]
        axf.plot(xs, ys, ":", color="#bbb", lw=0.8, zorder=1)
        axf.annotate(f"F1={f1v}", (0.42, f1v * 0.42 / (2 * 0.42 - f1v)), fontsize=7, color="#999")
    seen_g = set()
    for _, _, _, _, _, g, col, mk in pts:
        if g not in seen_g:
            seen_g.add(g)
            axf.scatter([], [], c=col, marker=mk, s=70, label=g)
    axf.set_xlabel("test Precision"); axf.set_ylabel("test Recall")
    axf.set_xlim(0.55, 0.95); axf.set_ylim(0.55, 1.02)
    axf.set_title("[그림 9f] 전 22개 config · test Precision–Recall  (점선 = F1 등고선)", fontsize=10)
    axf.legend(fontsize=8, loc="lower left"); axf.grid(alpha=0.2)
    figf.savefig(ROOT / "fig9f_frontier.png", dpi=140, bbox_inches="tight")
    print(f"→ {ROOT / 'fig9f_frontier.png'}")

    # 콘솔 표
    print("\nset                              devτ*  devF1  tP    tR    tF1   ΔvsA   selfF1  band428  안정")
    for name, slug in all_sets:
        m = mets[slug]
        print(f"  {name:<31} {m['tau_star']:.2f}   {m['dev']['F1']:.3f}  "
              f"{m['test']['P']:.2f}  {m['test']['R']:.2f}  {m['test']['F1']:.3f} "
              f"{m['test']['F1'] - a_f1:+.3f}  {m['test']['sweep_F1_max']:.3f}   "
              f"{m['undecidable_band']['all428']['frac']:.1%}  "
              f"{'불안정' if m['stability']['unstable'] else '안정'}")


if __name__ == "__main__":
    main()
