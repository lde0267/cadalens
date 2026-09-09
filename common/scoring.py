#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
τ 스윕 · confusion · P/R/F1 · 판정불가 밴드 — 채점 로직 한 곳.

`02_experiments/prompt_ablation/run_ablation.py` 와
`01_preprocess/scripts/labeling/finalize_labels.py`, `03_pipeline/stages/evaluate.py`
가 공유한다.

**규약**: score = p_forest (또는 p_farm). "형질변경/built = positive". 기본 예측 규칙은
`score < tau -> positive` (`predict="below"`). farm/built 처럼 p_positive 를 직접 쓰면
`predict="above"`.
"""
from __future__ import annotations

import csv
from pathlib import Path

TAU_LO, TAU_HI, TAU_STEP = 0.50, 0.99, 0.01
PLATEAU_EPS = 0.01   # dev F1 곡선이 평평 → 최대치 ±EPS τ 들의 중앙값 (dev 노이즈 과적합 완화)


def read_csv(p: str | Path) -> list[dict]:
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def taus(lo: float = TAU_LO, hi: float = TAU_HI, step: float = TAU_STEP) -> list[float]:
    n = round((hi - lo) / step)
    return [round(lo + i * step, 2) for i in range(n + 1)]


def _pred_pos(score: float, tau: float, predict: str) -> bool:
    return score < tau if predict == "below" else score >= tau


def confusion(pairs: list[tuple[float, bool]], tau: float, predict: str = "below") -> dict:
    """pairs = [(score, gt_is_positive)]."""
    tp = fp = fn = tn = 0
    for score, gt_pos in pairs:
        pp = _pred_pos(score, tau, predict)
        tp += gt_pos and pp
        fp += (not gt_pos) and pp
        fn += gt_pos and (not pp)
        tn += (not gt_pos) and (not pp)
    return {"tau": tau, **prf1(tp, fp, fn, tn)}


def prf1(tp: int, fp: int, fn: int, tn: int) -> dict:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    return {"P": round(prec, 4), "R": round(rec, 4), "F1": round(f1, 4),
            "Acc": round(acc, 4), "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def sweep_tau(pairs, predict: str = "below", **kw) -> list[dict]:
    return [confusion(pairs, t, predict) for t in taus(**kw)]


def best_tau(pairs, predict: str = "below", plateau_eps: float = PLATEAU_EPS, **kw) -> dict:
    """F1 최대 근방(±plateau_eps) τ 들의 (하위)중앙값. confusion + f1_max + plateau_taus."""
    cand = sweep_tau(pairs, predict, **kw)
    f1_max = max(c["F1"] for c in cand)
    band = sorted((c for c in cand if c["F1"] >= f1_max - plateau_eps - 1e-9),
                  key=lambda c: c["tau"])
    mid = dict(band[(len(band) - 1) // 2])
    mid["f1_max"] = round(f1_max, 4)
    mid["plateau_taus"] = [c["tau"] for c in band]
    return mid


def band_frac(scores: list[float], tau: float, delta: float) -> tuple[int, int, float]:
    """|score − tau| ≤ delta 인 표본 수 / 전체 / 비율 (판정불가 회색지대)."""
    n = sum(1 for v in scores if abs(v - tau) <= delta)
    tot = len(scores)
    return n, tot, (round(n / tot, 4) if tot else 0.0)


def score_labels(gt: dict[str, str], pred_pos: dict[str, bool],
                 pos_label: str, hold_label: str = "보류") -> dict:
    """단일 임계 예측(pred_pos: pnu→bool) 을 손라벨(gt: pnu→라벨)로 채점.
    hold_label · '제외' · pred 에 없는 pnu 는 제외."""
    tp = fp = fn = tn = 0
    for pnu, g in gt.items():
        if g == hold_label or g == "제외" or pnu not in pred_pos:
            continue
        gp = (g == pos_label)
        pp = pred_pos[pnu]
        tp += gp and pp
        fp += (not gp) and pp
        fn += gp and (not pp)
        tn += (not gp) and (not pp)
    return prf1(tp, fp, fn, tn)
