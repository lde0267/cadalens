#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage 1 — 칩 확인/생성.

config 의 image.chip_mode + target.jimok 에 맞는 칩 폴더가
01_preprocess/chips/<sheet>_<mode>/ 에 있고 그 지목을 덮으면 재사용,
없으면 01_preprocess/scripts/s3_make_chips.py 를 서브프로세스로 호출해 만든다.
(칩 산출물은 전처리 폴더에 귀속 — 파이프라인 run 폴더에는 경로만 기록.)
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from common.paths import ROOT, chip_dir


def _covers(index_csv: Path, jimoks: set[str]) -> bool:
    if not index_csv.exists():
        return False
    with open(index_csv, encoding="utf-8-sig") as f:
        have = {r.get("jimok") for r in csv.DictReader(f)}
    return jimoks <= have


def run(cfg: dict, run_dir: Path) -> Path:
    jimoks = {j.strip() for j in cfg["target"]["jimok"].split(",") if j.strip()}
    mode = cfg["image"]["chip_mode"]
    sheet = cfg["target"].get("sheet", "37714092")
    cdir = chip_dir(mode, sheet)
    index = cdir / "index.csv"

    if _covers(index, jimoks):
        print(f"[chips] 재사용: {cdir}  (지목 {sorted(jimoks)} 포함)")
    else:
        s3_jimok = next(iter(jimoks)) if len(jimoks) == 1 else "all"
        print(f"[chips] 생성: {cdir}  (지목 {s3_jimok}, mode={mode})")
        cmd = [sys.executable, str(ROOT / "01_preprocess" / "scripts" / "s3_make_chips.py"),
               "--jimok", s3_jimok, "--mode", mode, "--limit", "0", "--outdir", str(cdir)]
        subprocess.run(cmd, check=True, cwd=ROOT)

    (run_dir / "1_chips.txt").write_text(str(cdir) + "\n", encoding="utf-8")
    print(f"[chips] → {run_dir / '1_chips.txt'}  ({cdir})")
    return cdir
