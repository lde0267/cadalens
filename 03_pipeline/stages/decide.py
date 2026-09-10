#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage 3 — 점수 → 개략 라벨 (참고용 뷰).

파이프라인의 최종 산출은 stage 5 (`5_worklist.csv`, 우선순위 큐) 다. 이 단계는
임계값 하나로 자른 개략적인 O/X 와 몽타주를 참고용으로 남긴다.

config.decision.mode:
  binary   : score >= tau        → 정상(임야),  아니면 형질변경의심
  farmland : score >= built_tau  → 형질변경의심 · gh_sim 로 비닐하우스검토 · 그 외 농지
config.decision.norm ∈ {none, zscore, rank}: score 를 run 내 정규화 후 전역 norm_tau 로 자름.
2차 트리거: change_frac >= frac_tau (부분 전환).  ※ 실측상 τ 는 도엽마다 흔들려 신뢰 낮음.

산출: <run_dir>/3_labels.csv · 3_ranked.png · 3_summary.json
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from common.scoring import read_csv, normalize


def _thr(d: dict, raw_key: str) -> float:
    """norm 이 켜져 있으면 norm_tau, 아니면 원점수 임계값(tau/built_tau)."""
    if d.get("norm", "none") != "none":
        return float(d.get("norm_tau", d.get(raw_key, 0.9)))
    return float(d.get(raw_key, 0.9))


def _label_binary(r: dict, d: dict) -> str:
    thr = _thr(d, "tau")
    frac_tau = d.get("frac_tau", 1.01)          # 기본 off — config 에 주면 2차 트리거
    normal = d.get("normal_label", "임야")
    pos = d.get("positive_label", "형질변경의심")
    if float(r["_score"]) < thr:
        return pos
    if float(r.get("change_frac", 0) or 0) >= frac_tau:   # 전체는 '숲'이라도 패치 상당수가 전환
        return pos
    return normal


def _label_farmland(r: dict, d: dict) -> str:
    thr = _thr(d, "built_tau")
    gh_min = d.get("gh_min", 0.24)
    gh_margin = d.get("gh_margin", 0.03)
    normal = d.get("normal_label", "농지")
    pos = d.get("positive_label", "형질변경의심")
    gh_label = d.get("greenhouse_label", "비닐하우스검토")
    frac_tau = d.get("frac_tau", 1.01)
    if float(r["_score"]) >= thr:
        return pos
    if float(r.get("change_frac", 0) or 0) >= frac_tau:
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
    norm = d.get("norm", "none")
    pos_label = d.get("positive_label", "형질변경의심")
    scores = read_csv(run_dir / "2_scores.csv")

    raw_col = "p_screen" if mode == "farmland" else "p_forest"
    ns = normalize([float(r[raw_col] or 0) for r in scores], norm)
    for r, s in zip(scores, ns):
        r["_score"] = round(s, 4)
        r["norm_score"] = r["_score"]
    if norm != "none":
        print(f"[decide] norm={norm}  임계 norm_tau={d.get('norm_tau')}  ({raw_col} 도엽내 정규화)")

    labeler = _label_farmland if mode == "farmland" else _label_binary
    for r in scores:
        r["label"] = labeler(r, d)

    out = run_dir / "3_labels.csv"
    cols = ["pnu", "jibun", "jimok", "area_m2", "p_forest", "p_screen", "norm_score",
            "change_frac", "change_frac50", "n_patch",
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
