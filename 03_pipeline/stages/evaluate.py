#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage 4 (선택) — 손라벨이 있으면 채점.

config.eval.labels (pnu,label) + config.eval.split (pnu,split) 이 있으면:
  dev 에서 τ 스윕 → τ*  ·  test 를 τ* 로 1회 채점  ·  판정불가 밴드  ·  안정성
없으면 전체를 대상으로 τ 스윕만.

positive = 형질변경.  binary 는 p_forest < τ → positive,
farmland 는 p_screen >= τ → positive (predict='above').

산출: <run_dir>/4_eval.json
"""
from __future__ import annotations

import json
from pathlib import Path

from common.scoring import read_csv, confusion, best_tau, band_frac

POS = "형질변경"


def run(cfg: dict, run_dir: Path) -> Path | None:
    ev = cfg.get("eval") or {}
    labels_p = ev.get("labels")
    if not labels_p or not Path(labels_p).exists():
        print("[eval] config.eval.labels 없음 — 건너뜀")
        return None

    mode = cfg["decision"].get("mode", "binary")
    predict = "below" if mode != "farmland" else "above"
    score_col = "p_forest" if mode != "farmland" else "p_screen"

    scores = {r["pnu"]: r for r in read_csv(run_dir / "2_scores.csv")}
    for r in scores.values():
        r["_s"] = float(r[score_col])
    labels = {r["pnu"]: r["label"] for r in read_csv(labels_p)}
    all_s = [r["_s"] for r in scores.values()]

    split_p = ev.get("split")
    have_split = split_p and Path(split_p).exists()
    split = {r["pnu"]: r["split"] for r in read_csv(split_p)} if have_split else {}

    subsets = {"dev": [], "test": [], "all": []}
    miss = 0
    for pnu, g in labels.items():
        if g in ("보류", "제외"):
            continue
        s = scores.get(pnu)
        if s is None:
            miss += 1
            continue
        pair = (s["_s"], g == POS)
        subsets["all"].append(pair)
        if have_split:
            subsets.get(split.get(pnu, ""), []).append(pair)

    delta = ev.get("delta", 0.05)
    out = {"score_col": score_col, "predict": predict, "n_missing": miss}

    if have_split and subsets["dev"] and subsets["test"]:
        dev_best = best_tau(subsets["dev"], predict=predict)
        tau_star = dev_best["tau"]
        test_at = confusion(subsets["test"], tau_star, predict=predict)
        test_best = best_tau(subsets["test"], predict=predict)
        b_all = band_frac(all_s, tau_star, delta)
        b_test = band_frac([p for p, _ in subsets["test"]], tau_star, delta)
        out.update({
            "tau_star": tau_star,
            "dev_at_tau_star": confusion(subsets["dev"], tau_star, predict=predict),
            "test_at_tau_star": test_at,
            "test_self_best": test_best,
            "stable": abs(tau_star - test_best["tau"]) <= 0.10,
            "undecidable_band": {"delta": delta,
                                 "all": {"n": b_all[0], "total": b_all[1], "frac": b_all[2]},
                                 "test": {"n": b_test[0], "total": b_test[1], "frac": b_test[2]}},
        })
        print(f"[eval] τ*={tau_star}  test P={test_at['P']:.3f} R={test_at['R']:.3f} "
              f"F1={test_at['F1']:.3f}  ({'안정' if out['stable'] else '불안정'})")
    else:
        full = best_tau(subsets["all"], predict=predict)
        out.update({"full_sweep_best": full,
                    "at_config_tau": confusion(
                        subsets["all"],
                        cfg["decision"].get("tau" if predict == "below" else "built_tau", full["tau"]),
                        predict=predict)})
        print(f"[eval] (split 없음) 전체 best τ={full['tau']} F1={full['F1']:.3f}")

    (run_dir / "4_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] → {run_dir / '4_eval.json'}")
    return run_dir / "4_eval.json"
