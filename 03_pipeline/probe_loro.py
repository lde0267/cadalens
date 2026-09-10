#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LORO 선형프로브 — frozen RemoteCLIP 피처 위에 로지스틱 회귀, Leave-One-Region-Out.

각 지목군(imya/nongji) 마다:
  피처  = 서브프롬프트별 코사인(정상 N + 형질변경 M) + [p, change_frac, change_frac50,
          n_patch, valid_ratio, log(area)] (+ gh_sim)
  라벨  = 03_pipeline/data/<sheet>_<group>/labels.json  (형질변경 vs 정상, 제외·보류 제외)
  LORO  = 도엽 하나 빼고 학습 → 뺀 도엽 예측, 전 도엽 회전.

산출: runs/_probe_loro.csv  +  표
  - 프로브 held-out AUC/AP
  - 프로브 F1 @ 전역 τ=0.5 (배포 가능)  vs  프로브 F1 @ 그 도엽 최적 τ (오라클)
  - zero-shot 대비 (4_eval.json 의 full_sweep_best = zero-shot 오라클)

    python 03_pipeline/probe_loro.py
"""
from __future__ import annotations

import csv
import json
import math
import os
from glob import glob
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

PIPE = Path(__file__).resolve().parent
RUNS = PIPE / "runs"
DATA = PIPE / "data"


def prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def best_f1(y, p):
    o = np.argsort(-p)
    ys = y[o]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    fn = ys.sum() - tp
    P = tp / np.maximum(tp + fp, 1)
    R = tp / np.maximum(tp + fn, 1)
    F = 2 * P * R / np.maximum(P + R, 1e-9)
    i = int(np.argmax(F))
    return float(F[i]), float(p[o][i]), float(P[i]), float(R[i])


def load_group(group: str):
    """반환: X, y, regions(list), feat_names."""
    rows_all, y_all, reg_all = [], [], []
    for run in sorted(RUNS.glob(f"*_{group}")):
        sheet = run.name.split("_")[0]
        sc_p = run / "2_scores.csv"
        lab_p = DATA / run.name / "labels.json"
        if not sc_p.exists() or not lab_p.exists():
            continue
        sc = {r["pnu"]: r for r in csv.DictReader(open(sc_p, encoding="utf-8-sig"))}
        for r in json.load(open(lab_p, encoding="utf-8"))["labels"]:
            if r["label"] in ("제외", "보류") or r["pnu"] not in sc:
                continue
            s = sc[r["pnu"]]
            if "sims_normal" not in s or not s["sims_normal"]:
                raise SystemExit("2_scores.csv 에 sims_normal 없음 — classify 재실행 필요 "
                                 "(python 03_pipeline/eval_regions.py)")
            sn = [float(x) for x in s["sims_normal"].split()]
            sd = [float(x) for x in s["sims_changed"].split()]
            base = [
                float(s.get("p_forest") or s.get("p_screen") or 0),
                float(s.get("change_frac") or 0),
                float(s.get("change_frac50") or 0),
                float(s.get("n_patch") or 0),
                float(s.get("valid_ratio") or 0),
                math.log(float(s.get("area_m2") or 1) + 1),
            ]
            if s.get("gh_sim"):
                base.append(float(s["gh_sim"]))
            rows_all.append(sn + sd + base)
            y_all.append(1 if r["label"] == "형질변경" else 0)
            reg_all.append(sheet)
    X = np.array(rows_all, float)
    y = np.array(y_all, int)
    nS = len(sn)
    nD = len(sd)
    names = ([f"norm{i}" for i in range(nS)] + [f"chg{i}" for i in range(nD)]
             + ["p", "change_frac", "change_frac50", "n_patch", "valid_ratio", "log_area"]
             + (["gh_sim"] if X.shape[1] > nS + nD + 6 else []))
    return X, y, np.array(reg_all), names


def zeroshot_f1(sheet: str, group: str):
    p = RUNS / f"{sheet}_{group}" / "4_eval.json"
    if not p.exists():
        return None
    b = json.loads(p.read_text(encoding="utf-8")).get("full_sweep_best", {})
    return b.get("F1")


def run_group(group: str, out_rows: list):
    X, y, reg, names = load_group(group)
    regions = sorted(set(reg))
    print(f"\n===== {group}  (필지 {len(y)}, 양성 {y.sum()} = {y.mean():.0%}, 도엽 {len(regions)}, 피처 {X.shape[1]}) =====")
    print(f"{'도엽':10} {'N':>5} {'prev':>5} | {'AUC':>5} {'AP':>5} | "
          f"{'F1@0.5':>7} {'P':>5} {'R':>5} | {'F1@best':>8} {'(τ)':>6} | {'0shot F1':>8} {'Δ(best)':>8}")
    oof_p = np.zeros(len(y))
    for r in regions:
        te = reg == r
        tr = ~te
        sca = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(sca.transform(X[tr]), y[tr])
        p = clf.predict_proba(sca.transform(X[te]))[:, 1]
        oof_p[te] = p
        yt = y[te]
        auc = roc_auc_score(yt, p) if 0 < yt.sum() < len(yt) else float("nan")
        ap = average_precision_score(yt, p) if yt.sum() else float("nan")
        pred = p >= 0.5
        tp = int((pred & (yt == 1)).sum()); fp = int((pred & (yt == 0)).sum()); fn = int((~pred & (yt == 1)).sum())
        P5, R5, F5 = prf(tp, fp, fn)
        Fb, tb, Pb, Rb = best_f1(yt, p)
        zs = zeroshot_f1(r, group)
        dz = (Fb - zs) if zs is not None else float("nan")
        print(f"{r:10} {te.sum():5d} {yt.mean():>5.2f} | {auc:>5.2f} {ap:>5.2f} | "
              f"{F5:>7.3f} {P5:>5.2f} {R5:>5.2f} | {Fb:>8.3f} {tb:>6.2f} | "
              f"{('%.3f'%zs) if zs is not None else '   -  ':>8} {dz:>+8.3f}")
        out_rows.append({"group": group, "region": r, "n": int(te.sum()), "prev": round(float(yt.mean()), 3),
                         "probe_auc": round(float(auc), 3), "probe_ap": round(float(ap), 3),
                         "probe_f1_0.5": round(F5, 3), "probe_P_0.5": round(P5, 3), "probe_R_0.5": round(R5, 3),
                         "probe_f1_best": round(Fb, 3), "probe_tau_best": round(tb, 3),
                         "zeroshot_f1_best": zs, "delta_best": round(dz, 3) if zs is not None else None})
    # pooled (out-of-fold)
    auc = roc_auc_score(y, oof_p); ap = average_precision_score(y, oof_p)
    pred = oof_p >= 0.5
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    P5, R5, F5 = prf(tp, fp, fn)
    Fb, tb, Pb, Rb = best_f1(y, oof_p)
    print(f"{'POOLED':10} {len(y):5d} {y.mean():>5.2f} | {auc:>5.2f} {ap:>5.2f} | "
          f"{F5:>7.3f} {P5:>5.2f} {R5:>5.2f} | {Fb:>8.3f} {tb:>6.2f} |")
    # 전 도엽 학습 → 피처 계수 (해석용)
    sca = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sca.transform(X), y)
    coef = sorted(zip(names, clf.coef_[0]), key=lambda t: -abs(t[1]))[:8]
    print("  주요 피처 |coef|:  " + " · ".join(f"{n}{c:+.2f}" for n, c in coef))
    out_rows.append({"group": group, "region": "POOLED", "n": len(y), "prev": round(float(y.mean()), 3),
                     "probe_auc": round(float(auc), 3), "probe_ap": round(float(ap), 3),
                     "probe_f1_0.5": round(F5, 3), "probe_P_0.5": round(P5, 3), "probe_R_0.5": round(R5, 3),
                     "probe_f1_best": round(Fb, 3), "probe_tau_best": round(tb, 3),
                     "zeroshot_f1_best": None, "delta_best": None})


def main():
    out_rows = []
    for g in ("imya", "nongji"):
        run_group(g, out_rows)
    RUNS.mkdir(exist_ok=True)
    with open(RUNS / "_probe_loro.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n→ {RUNS / '_probe_loro.csv'}")


if __name__ == "__main__":
    main()
