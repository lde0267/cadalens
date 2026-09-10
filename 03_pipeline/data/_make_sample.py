#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""03_pipeline/data/ 샘플 입력 세트 생성 (세종 36710022 도엽 — 개발제한구역 검증용).

파이프라인 자체는 이 스크립트를 필요로 하지 않는다 — 입력 칩 폴더(<pnu>.png + index.csv)와
(심각도 단계용) 필지 폴리곤 gpkg 만 있으면 run.py 가 돈다. 이 스크립트는 저장소의
01_preprocess 산출에서 소수를 골라 처음부터 끝까지(심각도 포함) 도는 샘플을 만든다.

    python 03_pipeline/data/_make_sample.py
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

PIPE = Path(__file__).resolve().parents[1]          # 03_pipeline/
ROOT = PIPE.parent
SRC_CHIPS = ROOT / "01_preprocess" / "chips"
SRC_GPKG = ROOT / "01_preprocess" / "derived" / "sejong_36710022_parcels_within.gpkg"
GREENBELT = PIPE / "greenbelt" / "LSMD_CONT_UD801_세종"

# (원본 칩 폴더, 샘플 폴더명, 뽑을 지목, 개수, GB섞기)
#   gb_mix=(n_out, n_in): 개발제한구역 밖 n_out + 안 n_in — 심각도 gradation 과 GB 규칙 둘 다 보이게.
SAMPLES = [
    ("36710022_imya_real",   "imya_sample",   ["임"],            None, (20, 20)),
    ("36710022_nongji_mask", "nongji_sample", ["전", "답", "과"], 45,   None),
]


def _gb_status() -> dict:
    """pnu → 개발제한구역과 겹치면 True."""
    try:
        import geopandas as gpd
    except ImportError:
        return {}
    shp = sorted(GREENBELT.glob("*.shp"))
    if not shp or not SRC_GPKG.exists():
        return {}
    gbu = gpd.read_file(shp[0]).to_crs(5186).geometry.union_all()
    g = gpd.read_file(SRC_GPKG).to_crs(5186)
    return {str(p): bool(geom.intersection(gbu).area >= 1.0)
            for p, geom in zip(g["PNU"], g.geometry) if geom is not None}


def build(src_name: str, dst_name: str, jimoks: list[str],
          n: int | None, gb_mix: tuple[int, int] | None, gb: dict) -> None:
    src = SRC_CHIPS / src_name
    idx = src / "index.csv"
    if not idx.exists():
        sys.exit(f"원본 index.csv 없음: {idx}")
    with open(idx, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())

    jset = set(jimoks)
    cand = [r for r in rows
            if (r.get("jimok") or "").strip() in jset and (src / r["chip"]).exists()]

    if gb_mix and gb:
        n_out, n_in = gb_mix
        outs = [r for r in cand if not gb.get(r["pnu"], False)]
        ins = [r for r in cand if gb.get(r["pnu"], False)]
        picked = outs[:n_out] + ins[:n_in]
    else:
        n = n or len(cand)
        per = max(1, n // len(jimoks))
        seen = {j: 0 for j in jimoks}
        picked = []
        for r in cand:                       # 1차: 지목 균등
            if seen[r["jimok"]] < per:
                picked.append(r)
                seen[r["jimok"]] += 1
            if len(picked) >= n:
                break
        if len(picked) < n:                  # 2차: 백필
            have = {r["chip"] for r in picked}
            picked += [r for r in cand if r["chip"] not in have][:n - len(picked)]

    dst = PIPE / "data" / dst_name
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.glob("*.png"):
        old.unlink()
    for r in picked:
        shutil.copyfile(src / r["chip"], dst / r["chip"])
    with open(dst / "index.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(picked)

    dist: dict[str, int] = {}
    for r in picked:
        dist[r["jimok"]] = dist.get(r["jimok"], 0) + 1
    n_in = sum(gb.get(r["pnu"], False) for r in picked) if gb else 0
    print(f"{dst_name:14s} {len(picked):3d}개  지목분포 {dist}  (GB 안 {n_in})  ← {src_name}")


if __name__ == "__main__":
    gb = _gb_status()
    if not gb:
        print("!! geopandas/gpkg 없음 — GB 섞기 생략, 일반 표집")
    for src_name, dst_name, jimoks, n, gb_mix in SAMPLES:
        build(src_name, dst_name, jimoks, n, gb_mix, gb)
    # 심각도 단계용 필지 폴리곤 (전체 도엽분 — run.py 가 PNU 로 조인)
    if SRC_GPKG.exists():
        shutil.copyfile(SRC_GPKG, PIPE / "data" / "sejong_36710022_parcels.gpkg")
        print(f"parcels.gpkg   {SRC_GPKG.name}  → data/sejong_36710022_parcels.gpkg")
    else:
        print(f"!! 폴리곤 gpkg 없음: {SRC_GPKG} — 심각도 단계는 config.severity.parcels 를 직접 지정")
    print("\n샘플 준비 완료 → 03_pipeline/data/")
