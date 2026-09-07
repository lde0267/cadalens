#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage 3 — 점수 → 라벨.

config.decision.mode:
  binary   : p_forest >= tau            → 정상(임야),  아니면 형질변경의심
  farmland : p_screen >= built_tau      → 형질변경의심
             gh_sim 이 farm/built 를 gh_margin 이상 웃돌면 → 비닐하우스검토
             그 외                       → 농지

산출: <run_dir>/3_labels.csv · 3_ranked.png · 3_summary.json
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from common.scoring import read_csv


def _label_binary(r: dict, d: dict) -> str:
    tau = d.get("tau", 0.9)
    normal = d.get("normal_label", "임야")
    pos = d.get("positive_label", "형질변경의심")
    return normal if float(r["p_forest"]) >= tau else pos


def _label_farmland(r: dict, d: dict) -> str:
    built_tau = d.get("built_tau", 0.60)
    gh_min = d.get("gh_min", 0.24)
    gh_margin = d.get("gh_margin", 0.03)
    normal = d.get("normal_label", "농지")
    pos = d.get("positive_label", "형질변경의심")
    gh_label = d.get("greenhouse_label", "비닐하우스검토")
    p_screen = float(r["p_screen"])
    if p_screen >= built_tau:
        return pos
    gh = float(r.get("gh_sim", 0) or 0)
    other = max(float(r["closest_pos_cos"] or 0), float(r["closest_neg_cos"] or 0))
    if gh >= gh_min and gh - other >= gh_margin:
        return gh_label
    return normal


def _montage(rows, chips_dir: Path, out_path: Path, key, reverse, pos_label, cols=8, cell=200, topn=60):
    rr = sorted(rows, key=lambda r: float(r[key]), reverse=reverse)[:topn]
    if not rr:
        return
    rn = (len(rr) + cols - 1) // cols
    cap = 18
    cv = Image.new("RGB", (cols * cell, rn * (cell + cap)), (26, 26, 26))
    dr = ImageDraw.Draw(cv)
    for i, r in enumerate(rr):
        p = chips_dir / r["chip"]
        if not p.exists():
            continue
        im = Image.open(p).convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (26, 26, 26, 255)), im)
        im = im.convert("RGB").resize((cell, cell))
        x, y = (i % cols) * cell, (i // cols) * (cell + cap)
        cv.paste(im, (x, y))
        dr.rectangle([x, y + cell, x + cell, y + cell + cap], fill=(0, 0, 0))
        val = float(r[key])
        dr.text((x + 3, y + cell + 3), f'{val:.2f} {r["closest_neg"][:22]}',
                fill=(255, 90, 90) if r["label"] == pos_label else (150, 200, 150))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv.save(out_path)


def run(cfg: dict, run_dir: Path, chips_dir: Path) -> Path:
    d = cfg["decision"]
    mode = d.get("mode", "binary")
    pos_label = d.get("positive_label", "형질변경의심")
    scores = read_csv(run_dir / "2_scores.csv")
    labeler = _label_farmland if mode == "farmland" else _label_binary
    for r in scores:
        r["label"] = labeler(r, d)

    out = run_dir / "3_labels.csv"
    cols = ["pnu", "jibun", "jimok", "area_m2", "p_forest", "p_screen",
            "label", "closest_neg", "closest_neg_cos", "valid_ratio"]
    if mode == "farmland":
        cols += ["gh_sim"]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(scores)

    counts = Counter(r["label"] for r in scores)
    flagged = [r for r in scores if r["label"] == pos_label]
    summary = {
        "mode": mode,
        "decision": d,
        "n": len(scores),
        "counts": counts.most_common(),
        "flagged_closest_neg": Counter(r["closest_neg"] for r in flagged).most_common(15),
    }
    (run_dir / "3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    key, rev = ("p_screen", True) if mode == "farmland" else ("p_forest", False)
    _montage(scores, chips_dir, run_dir / "3_ranked.png", key, rev, pos_label,
             topn=cfg.get("montage_n", 60))

    print(f"[decide] → {out}")
    print(f"[decide] → {run_dir / '3_summary.json'}   " +
          " · ".join(f"{k} {v}" for k, v in counts.most_common()))
    print(f"[decide] → {run_dir / '3_ranked.png'}")
    return out
