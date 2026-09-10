#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage 1 — 입력 칩 폴더 확인.

파이프라인은 **이미 만들어진 필지 칩 폴더**를 입력으로 받는다 (전처리 s1~s3 불필요).
config 의 `target.chips` 가 그 폴더를 가리키고, 여기서는

  - `index.csv` 가 있는지
  - config 의 `target.jimok` 지목이 index 에 들어있는지
  - `<...>.png` 칩이 실제로 있는지

만 확인한 뒤 경로를 run 폴더에 `1_chips.txt` 로 기록한다.

입력 칩 폴더 규약:
  <chips_dir>/
    index.csv          # 헤더에 최소 chip,pnu,jibun,jimok,area_m2,valid_ratio
    <pnu>_<jibun>.png  # RGBA (alpha = 폴리곤 마스크: 내부 255 / 외부 0)
"""
from __future__ import annotations

import csv
from pathlib import Path


def _index_jimoks(index_csv: Path) -> set[str]:
    with open(index_csv, encoding="utf-8-sig") as f:
        return {(r.get("jimok") or "").strip() for r in csv.DictReader(f)}


def run(cfg: dict, run_dir: Path) -> Path:
    jimoks = {j.strip() for j in cfg["target"]["jimok"].split(",") if j.strip()}
    cdir = Path(cfg["target"]["chips"])          # run.py 가 절대경로로 해결해 넣어줌
    index = cdir / "index.csv"

    if not cdir.is_dir():
        raise SystemExit(f"[chips] 입력 칩 폴더 없음: {cdir}\n"
                         f"        config 의 target.chips 를 <pnu>.png + index.csv 가 든 폴더로 지정하세요.")
    if not index.exists():
        raise SystemExit(f"[chips] index.csv 없음: {index}")

    have = _index_jimoks(index)
    missing = jimoks - have
    if missing:
        raise SystemExit(f"[chips] 지목 {sorted(missing)} 이 index.csv 에 없음 "
                         f"(있는 지목: {sorted(j for j in have if j)})")

    n_png = sum(1 for p in cdir.glob("*.png") if not p.name.startswith("_"))
    if n_png == 0:
        raise SystemExit(f"[chips] png 칩이 하나도 없음: {cdir}")

    (run_dir / "1_chips.txt").write_text(str(cdir) + "\n", encoding="utf-8")
    print(f"[chips] 입력: {cdir}  (지목 {sorted(jimoks)} 포함, png {n_png}개)")
    print(f"[chips] → {run_dir / '1_chips.txt'}")
    return cdir
