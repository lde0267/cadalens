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

from common.scoring import read_csv, confusion, best_tau, band_frac, normalize

POS = "형질변경"

# norm 모드별 τ 스윕 범위 (best_tau(**kw) → taus(lo,hi,step))
SWEEP = {"none": {}, "zscore": dict(lo=-3.0, hi=3.0, step=0.1),
         "z": dict(lo=-3.0, hi=3.0, step=0.1), "rank": dict(lo=0.02, hi=0.98, step=0.02)}


def _ranking(pairs, predict: str, run_dir: Path) -> dict | None:
    """우선순위 큐 관점 지표 — 임계값 없이 '위험순 정렬'을 평가.
    recall@budget · lift · 예산→목표recall · AUROC/Gini/AP · CAP 곡선 PNG."""
    import numpy as np
    from common.scoring import ap, roc_auc
    if not pairs:
        return None
    s = np.array([p for p, _ in pairs], float)
    y = np.array([1 if yy else 0 for _, yy in pairs])
    npos = int(y.sum())
    if npos == 0 or npos == len(y):
        return None
    risk = -s if predict == "below" else s               # 높을수록 위반 의심
    o = np.argsort(-risk, kind="mergesort")
    yy = y[o]
    n = len(yy)
    cum = np.cumsum(yy) / npos                            # 누적 recall
    frac = (np.arange(n) + 1) / n

    def rec_at(f):
        return float(cum[max(1, int(round(f * n))) - 1])

    def prec_at(f):
        k = max(1, int(round(f * n)))
        return float(yy[:k].sum() / k)

    def budget_to(r):
        return float((np.argmax(cum >= r) + 1) / n) if cum[-1] >= r else 1.0

    auc = roc_auc(s, y, predict)
    out = {
        "n": n, "n_pos": npos, "prevalence": round(npos / n, 4),
        "auroc": round(auc, 4), "gini": round(2 * auc - 1, 4), "ap": round(ap(s, y, predict), 4),
        "recall_at": {f"{int(f*100)}%": round(rec_at(f), 4) for f in (0.05, 0.10, 0.20, 0.30, 0.50)},
        "lift_at": {f"{int(f*100)}%": round(rec_at(f) / f, 3) for f in (0.10, 0.20, 0.30)},
        "precision_at": {f"{int(f*100)}%": round(prec_at(f), 4) for f in (0.10, 0.20)},
        "budget_to_recall": {"80%": round(budget_to(0.8), 4), "90%": round(budget_to(0.9), 4)},
    }
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        prev = npos / n
        fig, axp = plt.subplots(figsize=(4.2, 4.2))
        axp.plot(np.r_[0, frac], np.r_[0, cum], lw=2, label=f"model (AUC {auc:.2f})")
        axp.plot([0, 1], [0, 1], "--", c="gray", lw=1, label="random")
        axp.plot([0, prev, 1], [0, 1, 1], ":", c="green", lw=1, label="perfect")
        axp.axvline(0.20, c="orange", lw=0.8, ls="-")
        axp.set_xlabel("fraction of parcels reviewed (budget)")
        axp.set_ylabel("fraction of violations found (recall)")
        axp.set_title(f"{run_dir.name}  ·  rec@20%={rec_at(0.2):.2f}  lift={rec_at(0.2)/0.2:.1f}x")
        axp.legend(fontsize=8, loc="lower right")
        axp.grid(alpha=0.3)
        axp.set_xlim(0, 1)
        axp.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(run_dir / "4_cap.png", dpi=110)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[eval] CAP png 스킵: {e}")
    return out


def _load_labels(path: str) -> dict:
    """pnu → 라벨. 두 형식 지원:
      .csv   : 헤더에 pnu,label
      .json  : label_tool 지역 라벨 ({"labels": [{"pnu","label", ...}]})
    """
    p = Path(path)
    if p.suffix.lower() == ".json":
        d = json.loads(p.read_text(encoding="utf-8"))
        recs = d["labels"] if isinstance(d, dict) and "labels" in d else d
        return {r["pnu"]: r["label"] for r in recs}
    return {r["pnu"]: r["label"] for r in read_csv(path)}


def run(cfg: dict, run_dir: Path) -> Path | None:
    ev = cfg.get("eval") or {}
    labels_p = ev.get("labels")
    if not labels_p or not Path(labels_p).exists():
        print("[eval] config.eval.labels 없음 — 건너뜀")
        return None

    mode = cfg["decision"].get("mode", "binary")
    norm = cfg["decision"].get("norm", "none")
    predict = "below" if mode != "farmland" else "above"
    score_col = "p_forest" if mode != "farmland" else "p_screen"
    sweep_kw = SWEEP.get(norm, {})

    rows = read_csv(run_dir / "2_scores.csv")
    ns = normalize([float(r[score_col] or 0) for r in rows], norm)   # 도엽내 정규화 (decide 와 동일)
    for r, s in zip(rows, ns):
        r["_s"] = s
    scores = {r["pnu"]: r for r in rows}
    labels = _load_labels(labels_p)
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

    out["norm"] = norm

    if have_split and subsets["dev"] and subsets["test"]:
        dev_best = best_tau(subsets["dev"], predict=predict, **sweep_kw)
        tau_star = dev_best["tau"]
        test_at = confusion(subsets["test"], tau_star, predict=predict)
        test_best = best_tau(subsets["test"], predict=predict, **sweep_kw)
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
        full = best_tau(subsets["all"], predict=predict, **sweep_kw)
        if norm != "none":
            cfg_thr = cfg["decision"].get("norm_tau", full["tau"])
        else:
            cfg_thr = cfg["decision"].get("tau" if predict == "below" else "built_tau", full["tau"])
        out.update({"full_sweep_best": full,
                    "at_config_tau": confusion(subsets["all"], cfg_thr, predict=predict)})
        print(f"[eval] (split 없음{', norm=' + norm if norm != 'none' else ''}) "
              f"전체 best τ={full['tau']} F1={full['F1']:.3f}")

    out["ranking"] = _ranking(subsets["all"], predict, run_dir)
    if out["ranking"]:
        rk = out["ranking"]
        print(f"[eval] 랭킹: AUC {rk['auroc']:.3f} · rec@20% {rk['recall_at']['20%']:.2f} "
              f"(lift {rk['lift_at']['20%']:.1f}x) · 예산→80%회수 {rk['budget_to_recall']['80%']:.0%}")

    (run_dir / "4_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] → {run_dir / '4_eval.json'}")
    return run_dir / "4_eval.json"
