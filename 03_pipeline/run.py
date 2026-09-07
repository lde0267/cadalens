#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CadaLens 파이프라인 러너 — JSON 설정 하나로 이미지 처리 방식·프롬프트를 갈아끼우고,
단계별로 실행해 중간 산출물을 확인한다.

    python 03_pipeline/run.py --config 03_pipeline/configs/imya_precision.json --stage all
    python 03_pipeline/run.py --config .../imya_precision.json --stage classify   # 그 run 폴더에 재생성
    python 03_pipeline/run.py --config .../imya_precision.json --from classify --to eval

단계: chips → classify → decide → eval
run 폴더: 03_pipeline/runs/<config 파일이름>/   (재실행 시 갱신, --fresh 는 타임스탬프 폴더)
  1_chips.txt · 2_scores.csv · 3_labels.csv · 3_ranked.png · 3_summary.json · 4_eval.json
  config.json  (동결 복사)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PIPELINE_RUNS, ROOT, chip_dir  # noqa: E402
from stages import chips, classify, decide, evaluate  # noqa: E402

STAGES = ["chips", "classify", "decide", "eval"]


def _resolve_paths(cfg: dict) -> dict:
    """config 안의 상대경로를 저장소 루트 기준 절대경로로."""
    if "prompt_file" in cfg:
        cfg["prompt_file"] = str((ROOT / cfg["prompt_file"]).resolve())
    ev = cfg.get("eval") or {}
    for k in ("labels", "split"):
        if ev.get(k):
            ev[k] = str((ROOT / ev[k]).resolve())
    return cfg


def load_config(p: Path) -> dict:
    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg.setdefault("target", {}).setdefault("jimok", "임")
    cfg["target"].setdefault("sheet", "37714092")
    cfg.setdefault("image", {}).setdefault("encoder", "cls")
    cfg["image"].setdefault("chip_mode", "mask")
    cfg["image"].setdefault("fill", "black")
    cfg.setdefault("decision", {}).setdefault("mode", "binary")
    if "prompt_file" not in cfg:
        sys.exit("config 에 prompt_file 이 필요합니다.")
    return _resolve_paths(cfg)


def stage_range(args) -> list[str]:
    if args.stage and args.stage != "all":
        return [args.stage]
    lo = STAGES.index(args.from_) if args.from_ else 0
    hi = STAGES.index(args.to) if args.to else len(STAGES) - 1
    return STAGES[lo:hi + 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--stage", choices=STAGES + ["all"], default="all")
    ap.add_argument("--from", dest="from_", choices=STAGES)
    ap.add_argument("--to", choices=STAGES)
    ap.add_argument("--fresh", action="store_true", help="타임스탬프 run 폴더로 새로 시작")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["cpu"] = args.cpu

    stem = args.config.stem
    run_dir = PIPELINE_RUNS / (f"{stem}_{_dt.datetime.now():%Y%m%d_%H%M%S}" if args.fresh else stem)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / "config.json")

    todo = stage_range(args)
    print(f"config : {args.config}")
    print(f"run 폴더: {run_dir}")
    print(f"단계    : {' → '.join(todo)}\n")

    chips_dir = chip_dir(cfg["image"]["chip_mode"], cfg["target"]["sheet"])

    for st in todo:
        if st == "chips":
            chips_dir = chips.run(cfg, run_dir)
        elif st == "classify":
            if not (run_dir / "1_chips.txt").exists() and "chips" not in todo:
                chips_dir = chips.run(cfg, run_dir)
            classify.run(cfg, run_dir, chips_dir, batch=args.batch)
        elif st == "decide":
            if not (run_dir / "2_scores.csv").exists():
                sys.exit("[decide] 2_scores.csv 없음 — 먼저 --stage classify")
            decide.run(cfg, run_dir, chips_dir)
        elif st == "eval":
            if not (run_dir / "2_scores.csv").exists():
                sys.exit("[eval] 2_scores.csv 없음 — 먼저 --stage classify")
            evaluate.run(cfg, run_dir)

    print(f"\n완료 → {run_dir}")


if __name__ == "__main__":
    main()
