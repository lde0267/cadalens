#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""프롬프트별 반응 요약 — 검증 6개 도엽 통합.

각 run 이 실제 사용한 prompt_file(config.json) 의 positive 그룹 문장을 p1..pN 으로 두고,
'형질변경의심'으로 플래그된 필지가 어느 문장에 가장 가까웠는지(closest_neg) 집계 +
그중 실제 위반 비율(정밀도)을 낸다.

    python 03_pipeline/prompt_breakdown.py --group imya
    python 03_pipeline/prompt_breakdown.py --group nongji
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PIPE = Path(__file__).resolve().parent
DATA = PIPE / "data"
RUNS = PIPE / "runs"
POS = "형질변경"
PILOT = "36710022_"

GLOSS = {
    # imya_v3.json  changed (13) — 검증 run 이 실제로 쓴 세트
    "rows of dark blue solar photovoltaic panels on a cleared slope": "태양광 패널",
    "a house or building with a fenced yard on leveled hillside ground": "건물+마당",
    "a large warehouse or factory shed with a bright metal roof": "창고/공장",
    "a wide paved road or machine-graded dirt track cutting straight across the slope": "도로/임도",
    "smooth machine-graded bare earth with tire tracks and a sharp cleared edge": "기계 정지 나지",
    "a logged patch of bare soil dotted with small cut tree stumps": "벌채 나지",
    "rows of white plastic greenhouse tunnels": "비닐하우스",
    "a plowed field or an orchard with fruit trees planted in rows on former forest land": "개간 밭/과수",
    "rows of low rounded bare or grassy burial mounds": "분묘",
    "a quarry or open pit of excavated bare earth, gravel and rock": "채석/토취",
    "an open dirt yard with parked vehicles, shipping containers or stacked materials": "야적장",
    "a graded construction site with concrete foundations or retaining walls": "건설 기초/옹벽",
    "an open field of grass or low weeds on ground where the forest was removed": "벌목 후 초지",
    # nongji_change.json  built (9)
    "a building or house with a roof standing in a field": "건물/주택",
    "a large warehouse or factory shed with a wide metal roof": "창고/공장",
    "a long livestock barn or animal shed with a metal roof": "축사",
    "rows of dark blue solar photovoltaic panels covering the field": "태양광 패널",
    "a paved or concrete-covered lot with parked vehicles": "포장 부지+주차",
    "an open yard with stored shipping containers, pipes or stacked building materials": "야적장",
    "a raised pad of piled fill soil or earth dumped over the field": "성토 단",
    "machine-graded bare ground with tire tracks prepared for construction": "건설 정지 나지",
    "an asphalt road or concrete driveway across the farmland": "도로/진입로",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["imya", "nongji"], required=True)
    args = ap.parse_args()
    grp = args.group

    regs = [p.name for p in sorted(DATA.iterdir())
            if p.is_dir() and (p / "labels.json").exists()
            and p.name.endswith(f"_{grp}") and not p.name.startswith(PILOT)]

    prompts = None
    agg = {}
    n_flag_total = 0
    used_files = set()
    for name in regs:
        sc, lb, cf = RUNS / name / "2_scores.csv", DATA / name / "labels.json", RUNS / name / "config.json"
        if not (sc.exists() and lb.exists() and cf.exists()):
            continue
        pfile = json.loads(cf.read_text(encoding="utf-8"))["prompt_file"]
        used_files.add(pfile)
        pf = json.loads((PIPE.parent / pfile).read_text(encoding="utf-8"))
        pl = pf["groups"][pf["positive_group"]]
        if prompts is None:
            prompts = pl
            agg = {f"p{i+1}": {"flag": 0, "tp": 0, "cos_sum": 0.0} for i in range(len(pl))}
        idx = {s: f"p{i+1}" for i, s in enumerate(pl)}

        gt = {r["pnu"]: r["label"] for r in json.loads(lb.read_text(encoding="utf-8"))["labels"]}
        with open(sc, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                flagged = (float(r["p_screen"] or 0) >= 0.60) if grp == "nongji" \
                    else (float(r["p_forest"] or 0) < 0.85)
                if not flagged:
                    continue
                key = idx.get(r["closest_neg"])
                if key is None:
                    continue
                n_flag_total += 1
                agg[key]["flag"] += 1
                agg[key]["cos_sum"] += float(r["closest_neg_cos"] or 0)
                if gt.get(r["pnu"]) == POS:
                    agg[key]["tp"] += 1

    print(f"[{grp}]  검증 {len(regs)}개 도엽 · prompt_file={sorted(used_files)} · 플래그 {n_flag_total}개\n")
    print(f"{'':4} {'유형':14} {'flagged':>8} {'비중':>6} {'실제위반':>8} {'정밀도':>7} {'평균cos':>8}")
    for i, s in enumerate(prompts):
        k = f"p{i+1}"
        a = agg[k]
        share = a["flag"] / n_flag_total if n_flag_total else 0
        prec = a["tp"] / a["flag"] if a["flag"] else 0
        cos = a["cos_sum"] / a["flag"] if a["flag"] else 0
        print(f"{k:4} {GLOSS.get(s, s[:14]):14} {a['flag']:>8} {share:>5.0%} "
              f"{a['tp']:>8} {prec:>6.0%} {cos:>8.3f}")


if __name__ == "__main__":
    main()
