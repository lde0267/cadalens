#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""03_pipeline/data/ 지역(도엽)별 검증 세트 구축.

01_preprocess/labels/labels/labels_<sheet>_<group>.json (label_tool 지역 손라벨) 마다
  data/<sheet>_<group>/
    index.csv     원본 칩 index 스키마, 라벨된 필지만 (PNU 로 원본 index 에서 해결 — 라벨
                  JSON 의 chip 파일명은 재-칩 이후 슬러그가 바뀌었을 수 있어 신뢰하지 않음)
    <pnu>_*.png   라벨된 필지 칩 사본 ('제외' 라벨은 채점에 안 쓰므로 png 는 복사 안 함)
    labels.json   원본 라벨 JSON 그대로 복사 ('제외' 포함 — 채점 시 evaluate 가 스킵)

파이프라인 실행(지역별):
    python 03_pipeline/run.py --config 03_pipeline/configs/imya.json --no-severity \\
        --chips data/<sheet>_imya --labels data/<sheet>_imya/labels.json \\
        --run-name <sheet>_imya --from classify --to eval

    python 03_pipeline/data/_make_regions.py            # 전체
    python 03_pipeline/data/_make_regions.py 36709030   # 특정 도엽만 (인자 = sheet 필터)
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

PIPE = Path(__file__).resolve().parents[1]              # 03_pipeline/
ROOT = PIPE.parent
LABELS_DIR = ROOT / "01_preprocess" / "labels" / "labels"
CHIPS_DIR = ROOT / "01_preprocess" / "chips"
SKIP_COPY = {"제외"}                                     # png 사본 제외 (채점 미사용)


def build(label_json: Path) -> None:
    d = json.loads(label_json.read_text(encoding="utf-8"))
    sheet, group, src_name = d["sheet"], d["group"], d["dir"]
    recs = d["labels"]
    src = CHIPS_DIR / src_name
    src_idx = src / "index.csv"
    if not src_idx.exists():
        print(f"!! {label_json.name}: 원본 칩 폴더 없음 {src_idx} — 건너뜀")
        return
    with open(src_idx, encoding="utf-8-sig") as f:
        by_pnu = {r["pnu"]: r for r in csv.DictReader(f)}
        fields = list(next(iter(by_pnu.values())).keys())

    dst = PIPE / "data" / f"{sheet}_{group}"
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.glob("*.png"):
        old.unlink()

    picked, miss, copied = [], 0, 0
    dist = Counter()
    for r in recs:
        dist[r["label"]] += 1
        row = by_pnu.get(r["pnu"])
        if row is None:
            miss += 1
            continue
        picked.append(row)
        if r["label"] not in SKIP_COPY:
            sp = src / row["chip"]
            if sp.exists():
                shutil.copyfile(sp, dst / row["chip"])
                copied += 1
    with open(dst / "index.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(picked)
    shutil.copyfile(label_json, dst / "labels.json")

    scor = sum(v for k, v in dist.items() if k not in ("제외", "보류"))
    print(f"{sheet}_{group:6s}  라벨 {len(recs):4d} {dict(dist)}  "
          f"→ index {len(picked)} · png {copied} · 채점대상 {scor}"
          + (f"  (PNU미매칭 {miss})" if miss else ""))


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    files = sorted(LABELS_DIR.glob("labels_*.json"))
    if only:
        files = [f for f in files if only in f.name]
    if not files:
        sys.exit(f"라벨 JSON 없음: {LABELS_DIR}" + (f" (필터 '{only}')" if only else ""))
    for f in files:
        build(f)
    print(f"\n완료 → 03_pipeline/data/  ({len(files)}개 지역)")
