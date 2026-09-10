#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""지역(도엽)별 검증 일괄 실행.

03_pipeline/data/<sheet>_<group>/ (labels.json 있는 폴더) 마다 group 에 맞는 config 로
classify → decide → eval 을 돌리고 (severity 제외), 각 runs/<sheet>_<group>/4_eval.json 을
모아 runs/_regions_eval.csv + 표로 출력한다.

    python 03_pipeline/eval_regions.py                 # 전체
    python 03_pipeline/eval_regions.py --only 36709030 # 특정 도엽
    python 03_pipeline/eval_regions.py --group imya --cpu
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PIPE = Path(__file__).resolve().parent
DATA = PIPE / "data"
RUNS = PIPE / "runs"
CONFIG = {"imya": PIPE / "configs" / "imya.json", "nongji": PIPE / "configs" / "nongji.json"}


def regions(only: str | None, group: str | None) -> list[tuple[str, str, Path]]:
    out = []
    for p in sorted(DATA.iterdir()):
        if not (p.is_dir() and (p / "labels.json").exists()):
            continue
        name = p.name
        grp = "nongji" if name.endswith("_nongji") else ("imya" if name.endswith("_imya") else None)
        if grp is None or (group and grp != group) or (only and only not in name):
            continue
        out.append((name, grp, p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="sheet 필터 (부분일치)")
    ap.add_argument("--group", choices=["imya", "nongji"])
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--stage-from", default="classify", help="시작 단계 (기본 classify)")
    args = ap.parse_args()

    todo = regions(args.only, args.group)
    if not todo:
        sys.exit("대상 지역 없음 — 먼저 python 03_pipeline/data/_make_regions.py")
    print(f"지역 {len(todo)}개: " + ", ".join(n for n, _, _ in todo) + "\n")

    rows = []
    for name, grp, path in todo:
        cmd = [sys.executable, str(PIPE / "run.py"), "--config", str(CONFIG[grp]),
               "--no-severity", "--chips", str(path), "--labels", str(path / "labels.json"),
               "--run-name", name, "--from", args.stage_from, "--to", "eval",
               "--batch", str(args.batch)]
        if args.cpu:
            cmd.append("--cpu")
        print(f"── {name} ({grp}) ──")
        r = subprocess.run(cmd, cwd=PIPE.parent)
        ev = RUNS / name / "4_eval.json"
        rec = {"region": name, "group": grp}
        if r.returncode == 0 and ev.exists():
            e = json.loads(ev.read_text(encoding="utf-8"))
            best = e.get("full_sweep_best") or e.get("test_at_tau_star") or {}
            at = e.get("at_config_tau") or {}
            rec.update({
                "n": (best.get("TP", 0) + best.get("FP", 0) + best.get("FN", 0) + best.get("TN", 0)),
                "best_tau": best.get("tau", ""), "best_P": best.get("P", ""),
                "best_R": best.get("R", ""), "best_F1": best.get("F1", ""),
                "cfg_tau": at.get("tau", ""), "cfg_P": at.get("P", ""),
                "cfg_R": at.get("R", ""), "cfg_F1": at.get("F1", ""),
                "n_missing": e.get("n_missing", ""),
            })
        else:
            rec["error"] = f"rc={r.returncode} eval={'있음' if ev.exists() else '없음'}"
        rows.append(rec)
        print()

    RUNS.mkdir(exist_ok=True)
    cols = ["region", "group", "n", "best_tau", "best_P", "best_R", "best_F1",
            "cfg_tau", "cfg_P", "cfg_R", "cfg_F1", "n_missing", "error"]
    with open(RUNS / "_regions_eval.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print("== 지역별 검증 요약 (config τ 기준 / 자체최적 τ) ==")
    print(f"{'region':18s} {'n':>5} {'cfg P/R/F1':>20} {'best τ P/R/F1':>24}")
    for r in rows:
        if "error" in r:
            print(f"{r['region']:18s}  {r['error']}")
            continue
        print(f"{r['region']:18s} {r['n']:>5} "
              f"{r['cfg_P']:>6}/{r['cfg_R']:>5}/{r['cfg_F1']:>5}   "
              f"τ{r['best_tau']:<4} {r['best_P']:>5}/{r['best_R']:>5}/{r['best_F1']:>5}")
    print(f"\n→ {RUNS / '_regions_eval.csv'}")


if __name__ == "__main__":
    main()
