#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CadaLens 파이프라인 러너 — 필지 칩 폴더를 넣으면 형질변경 의심 필지를 스크리닝한다.
JSON 설정 하나로 이미지 처리 방식·프롬프트를 갈아끼우고, 단계별로 실행해 중간 산출물을 확인한다.

    # 임야 파이프라인
    python 03_pipeline/run.py --config 03_pipeline/configs/imya.json --stage all --cpu
    # 농지 파이프라인
    python 03_pipeline/run.py --config 03_pipeline/configs/nongji.json --stage all --cpu

    python 03_pipeline/run.py --config .../imya.json --stage classify   # 그 run 폴더에 재생성
    python 03_pipeline/run.py --config .../imya.json --from classify --to eval

    # 지역(도엽)별 검증 — 하나의 config 를 칩·라벨만 갈아끼워 재사용
    python 03_pipeline/run.py --config .../imya.json --no-severity \
        --chips data/36709030_imya --labels data/36709030_imya/labels.json \
        --run-name 36709030_imya --from classify --to eval

단계: chips → classify → decide → severity → eval
  chips    = config 의 target.chips 칩 폴더 확인 (전처리 불필요, 이미 만들어진 칩을 입력)
  classify = RemoteCLIP zero-shot 2-way softmax + 패치별 형질변경비율 → 필지별 점수
  decide   = τ + frac_tau → 라벨 · 몽타주 · 요약
  severity = (선택) config.severity 있을 때 형질변경의심 필지 심각도
  eval     = (선택) eval.labels(.json/.csv) 있으면 τ 스윕 채점

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

PIPE_ROOT = Path(__file__).resolve().parent          # 03_pipeline/
sys.path.insert(0, str(PIPE_ROOT.parent))
from common.paths import ROOT  # noqa: E402
from stages import chips, classify, decide, evaluate, severity  # noqa: E402

STAGES = ["chips", "classify", "decide", "severity", "eval"]
RUNS = PIPE_ROOT / "runs"


def _resolve_chips(raw: str) -> str:
    """target.chips 상대경로를 03_pipeline/ → 저장소 루트 순으로 찾아 절대경로화."""
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    for base in (PIPE_ROOT, ROOT):
        if (base / p).exists():
            return str((base / p).resolve())
    return str((PIPE_ROOT / p).resolve())            # 없으면 03_pipeline 기준으로 리포트


def _resolve_paths(cfg: dict) -> dict:
    """config 안의 상대경로를 절대경로로."""
    cfg["target"]["chips"] = _resolve_chips(cfg["target"]["chips"])
    if "prompt_file" in cfg:
        cfg["prompt_file"] = str((ROOT / cfg["prompt_file"]).resolve())
    ev = cfg.get("eval") or {}
    for k in ("labels", "split"):
        if ev.get(k):
            ev[k] = _resolve_chips(ev[k])          # 03_pipeline/ → 루트 순으로 해결
    sv = cfg.get("severity") or {}
    for k in ("parcels", "greenbelt"):
        if sv.get(k):
            sv[k] = _resolve_chips(sv[k])
    return cfg


def load_config(p: Path, chips: str | None = None, labels: str | None = None) -> dict:
    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg.setdefault("target", {}).setdefault("jimok", "임")
    if chips:                                       # --chips 오버라이드
        cfg["target"]["chips"] = chips
    if labels:                                      # --labels 오버라이드 (지역별 검증)
        cfg.setdefault("eval", {})["labels"] = labels
    if not cfg["target"].get("chips"):
        sys.exit("config 의 target.chips (또는 --chips) 에 입력 칩 폴더 경로가 필요합니다 "
                 "(<pnu>.png + index.csv 가 든 폴더).")
    cfg.setdefault("image", {}).setdefault("encoder", "whole")
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
    ap.add_argument("--chips", help="config 의 target.chips 오버라이드 (칩 폴더)")
    ap.add_argument("--labels", help="config 의 eval.labels 오버라이드 (.json/.csv — 지역별 검증)")
    ap.add_argument("--severity-parcels", help="config 의 severity.parcels 오버라이드 (필지 gpkg)")
    ap.add_argument("--severity-greenbelt", help="config 의 severity.greenbelt 오버라이드 (.shp/폴더). 'none' 이면 비활성")
    ap.add_argument("--run-name", help="run 폴더 이름 (기본: config 파일이름). 지역별 검증 시 도엽별로 분리")
    ap.add_argument("--no-severity", action="store_true", help="severity 단계 건너뛰기")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    cfg = load_config(args.config, chips=args.chips, labels=args.labels)
    cfg["cpu"] = args.cpu
    if args.severity_parcels:
        cfg.setdefault("severity", {})["parcels"] = _resolve_chips(args.severity_parcels)
    if args.severity_greenbelt:
        sv = cfg.setdefault("severity", {})
        sv["greenbelt"] = None if args.severity_greenbelt.lower() == "none" \
            else _resolve_chips(args.severity_greenbelt)

    stem = args.run_name or args.config.stem
    run_dir = RUNS / (f"{stem}_{_dt.datetime.now():%Y%m%d_%H%M%S}" if args.fresh else stem)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / "config.json")

    todo = [s for s in stage_range(args) if not (args.no_severity and s == "severity")]
    print(f"config : {args.config}")
    print(f"입력 칩: {cfg['target']['chips']}")
    print(f"run 폴더: {run_dir}")
    print(f"단계    : {' → '.join(todo)}\n")

    chips_dir = Path(cfg["target"]["chips"])

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
        elif st == "severity":
            if not (run_dir / "2_scores.csv").exists():
                sys.exit("[severity] 2_scores.csv 없음 — 먼저 --stage classify")
            severity.run(cfg, run_dir, chips_dir)
        elif st == "eval":
            if not (run_dir / "2_scores.csv").exists():
                sys.exit("[eval] 2_scores.csv 없음 — 먼저 --stage classify")
            evaluate.run(cfg, run_dir)

    print(f"\n완료 → {run_dir}")


if __name__ == "__main__":
    main()
