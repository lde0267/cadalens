#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""세종시 전량 스크리닝 — 2 도엽 × 임야/농지 = 4 run → 임야/농지 우선순위 큐 2개.

각 run: chips → classify → decide → severity(전 필지 티어·연속 심각도).
36709030 은 개발제한구역 폴리곤이 없어 GB 항 = 0 (severity-greenbelt none).
36710022_nongji 는 손라벨이 없어 eval 없이 큐만.

산출:
  runs/sejong_<group>_worklist.csv   두 도엽 병합, group 내 (tier, risk) 재정렬 → 최종 rank
  runs/sejong_summary.json           티어·등급 분포 + (라벨 있는 run) 티어별 실제 형질변경 정밀도

    python 03_pipeline/sejong_run.py
    python 03_pipeline/sejong_run.py --skip-run     # 이미 돈 run 폴더로 병합만
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

PIPE = Path(__file__).resolve().parent
ROOT = PIPE.parent
RUNS = PIPE / "runs"
GB = "greenbelt/LSMD_CONT_UD801_세종"

# (run 이름, config, 칩 폴더, 라벨(.json|None), severity gpkg, greenbelt)
JOBS = [
    ("sejong_36709030_imya",   "imya.json",   "data/36709030_imya",   "data/36709030_imya/labels.json",   "data/sejong_36709030_parcels.gpkg", "none"),
    ("sejong_36709030_nongji", "nongji.json", "data/36709030_nongji", "data/36709030_nongji/labels.json", "data/sejong_36709030_parcels.gpkg", "none"),
    ("sejong_36710022_imya",   "imya.json",   "data/36710022_imya",   "data/36710022_imya/labels.json",   "data/sejong_36710022_parcels.gpkg", GB),
    ("sejong_36710022_nongji", "nongji.json", "data/36710022_nongji", None,                               "data/sejong_36710022_parcels.gpkg", GB),
]
TIER_ORDER = {"T1": 0, "T2": 1, "T3": 2}


def do_run(name, cfg, chips, labels, gpkg, gb, stage="all"):
    cmd = [sys.executable, str(PIPE / "run.py"), "--config", str(PIPE / "configs" / cfg),
           "--chips", chips, "--run-name", name, "--stage", stage,
           "--severity-parcels", gpkg, "--severity-greenbelt", gb]
    if labels:
        cmd += ["--labels", labels]
    print(f"\n══ {name} ({stage}) ══")
    subprocess.run(cmd, cwd=ROOT, check=True)


def label_map(p):
    fp = PIPE / p if p else None
    if not fp or not fp.exists():
        return {}
    d = json.loads(fp.read_text(encoding="utf-8"))
    return {r["pnu"]: r["label"] for r in d["labels"]}


def merge(group, jobs):
    rows = []
    for name, _, _, labels, _, _ in jobs:
        sev = RUNS / name / "5_worklist.csv"
        if not sev.exists():
            print(f"!! {sev} 없음 — 스킵")
            continue
        sheet = name.split("_")[1]
        lm = label_map(labels)
        for r in csv.DictReader(open(sev, encoding="utf-8-sig")):
            r["sheet"] = sheet
            r["label"] = lm.get(r["pnu"], "")
            rows.append(r)
    rows.sort(key=lambda r: (TIER_ORDER.get(r["tier"], 9), -float(r["risk"])))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    cols = ["rank", "sheet", "tier", "risk", "pnu", "jibun", "jimok", "area_m2", "closest_neg",
            "change_frac", "change_area_m2", "in_greenbelt", "gb_ratio",
            "severity_score", "severity_level", "label", "chip"]
    out = RUNS / f"sejong_{group}_worklist.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"→ {out}  ({len(rows)}필지)")
    return rows


def stats(group, rows):
    tc = Counter(r["tier"] for r in rows)
    lc = Counter(r["severity_level"] for r in rows)
    labeled = [r for r in rows if r["label"] in ("형질변경", "임야", "농지")]
    s = {"group": group, "n": len(rows),
         "tiers": {t: tc.get(t, 0) for t in ("T1", "T2", "T3")},
         "levels": {lv: lc.get(lv, 0) for lv in ("높음", "중간", "낮음")},
         "n_greenbelt": sum(1 for r in rows if r["in_greenbelt"] == "True")}
    if labeled:
        pos = sum(1 for r in labeled if r["label"] == "형질변경")
        s["labeled"] = {"n": len(labeled), "형질변경": pos, "prevalence": round(pos / len(labeled), 3)}
        for t in ("T1", "T2", "T3"):
            sub = [r for r in labeled if r["tier"] == t]
            if sub:
                p = sum(1 for r in sub if r["label"] == "형질변경")
                s.setdefault("tier_precision", {})[t] = {"n": len(sub), "형질변경": p,
                                                         "precision": round(p / len(sub), 3)}
        # 티어 정렬 순서로 recall@budget
        import numpy as np
        y = np.array([1 if r["label"] == "형질변경" else 0 for r in labeled])
        cum = np.cumsum(y) / max(y.sum(), 1)
        n = len(y)
        s["recall_at_budget"] = {f"{int(f*100)}%": round(float(cum[max(1, int(f * n)) - 1]), 3)
                                 for f in (0.1, 0.2, 0.3, 0.5)}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-run", action="store_true", help="run 은 건너뛰고 병합만")
    ap.add_argument("--severity-only", action="store_true", help="classify 재사용, severity 만 재실행")
    args = ap.parse_args()
    if not args.skip_run:
        for j in JOBS:
            do_run(*j, stage="severity" if args.severity_only else "all")
    summ = []
    for group in ("imya", "nongji"):
        jobs = [j for j in JOBS if j[0].endswith(group)]
        rows = merge(group, jobs)
        summ.append(stats(group, rows))
    (RUNS / "sejong_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(f"\n→ {RUNS / 'sejong_summary.json'}")
    for s in summ:
        print(f"\n[{s['group']}] {s['n']}필지  티어 {s['tiers']}  등급 {s['levels']}  GB {s['n_greenbelt']}")
        if "tier_precision" in s:
            print(f"  라벨 {s['labeled']}  티어별 실제 형질변경 정밀도 {s['tier_precision']}")
            print(f"  티어정렬 recall@budget {s['recall_at_budget']}")


if __name__ == "__main__":
    main()
