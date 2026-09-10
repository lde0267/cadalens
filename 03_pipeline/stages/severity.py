#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage 5 — 심각도 + 우선순위 큐 (임계값 없음, 전 필지).

`2_scores.csv` 의 모든 필지에 대해 두 축을 산출한다:

  risk (신뢰도)  = 1 - p_forest (binary) / p_screen (farmland).  형질변경일 가능성.
  severity (심각도, 0~100 연속)  — 순위엔 안 쓰이고 컬럼으로만 표시
      = w_gb · [개발제한구역 저촉]
      + w_cf · min(change_frac / cf_ref, 1)
      + w_area · clip( (log(전환면적) - log(area_lo)) / (log(area_ref) - log(area_lo)), 0, 1 )
    전환면적 = change_frac × 필지면적.  개발제한구역·필지 폴리곤 없으면 각 항 0.
  severity_level = score_bands [b_lo, b_hi] → 낮음 / 중간 / 높음.

정렬 = 사전식 (tier, -risk):
  T1  개발제한구역 저촉 & risk ≥ risk_floor
  T2  (T1 아님) & risk ≥ risk_floor & (면적 ≥ area_big_m2  또는  change_frac ≥ cf_hi)
  T3  나머지
  → T1(risk순) + T2(risk순) + T3(risk순) 이어붙여 rank 1..N.

config.severity (전부 선택, 기본값 있음):
  { "parcels": "...gpkg"(PNU),  "greenbelt": "...shp|dir",
    "risk_floor": 0.5, "area_big_m2": 30000(임야)/5000(농지), "cf_hi": 0.35,
    "w_gb": 40, "w_cf": 30, "w_area": 30, "cf_ref": 0.5,
    "area_lo_m2": 1000, "area_ref_m2": 20000, "score_bands": [30, 58] }

산출: <run_dir>/5_worklist.csv (전 필지 우선순위 큐) · 5_worklist.json · 5_worklist.png
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from common import geoenv  # noqa: F401  (PROJ/GDAL 격리 — geopandas 전에)
from common.scoring import read_csv

LEVELS = ["낮음", "중간", "높음"]


def _greenbelt_union(path: Path):
    import geopandas as gpd
    shp = path
    if path.is_dir():
        shps = sorted(path.glob("*.shp"))
        if not shps:
            raise SystemExit(f"[severity] greenbelt .shp 없음: {path}")
        shp = shps[0]
    gb = gpd.read_file(shp).to_crs(5186)
    return gb.geometry.union_all(), shp.name


def _parcel_geoms(path: Path) -> dict:
    import geopandas as gpd
    g = gpd.read_file(path).to_crs(5186)
    key = "PNU" if "PNU" in g.columns else ("pnu" if "pnu" in g.columns else None)
    if key is None:
        raise SystemExit(f"[severity] parcels 에 PNU 컬럼 없음: {path} ({list(g.columns)})")
    return {str(k): geom for k, geom in zip(g[key], g.geometry) if geom is not None}


def _montage(rows, chips_dir: Path, out_path: Path, cols=8, cell=200, topn=64):
    rr = rows[:topn]
    if not rr:
        return
    rn = (len(rr) + cols - 1) // cols
    cap = 20
    cv = Image.new("RGB", (cols * cell, rn * (cell + cap)), (26, 26, 26))
    dr = ImageDraw.Draw(cv)
    colour = {"높음": (255, 90, 90), "중간": (240, 190, 90), "낮음": (150, 200, 150)}
    for i, r in enumerate(rr):
        p = chips_dir / r["chip"] if r["chip"] else None
        if not p or not p.exists():
            continue
        im = Image.open(p).convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (26, 26, 26, 255)), im)
        im = im.convert("RGB").resize((cell, cell))
        x, y = (i % cols) * cell, (i // cols) * (cell + cap)
        cv.paste(im, (x, y))
        dr.rectangle([x, y + cell, x + cell, y + cell + cap], fill=(0, 0, 0))
        gb = " GB" if r["in_greenbelt"] else ""
        dr.text((x + 3, y + cell + 4),
                f'{r["tier"]} r{r["risk"]:.2f} s{r["severity_score"]:.0f}{gb}',
                fill=colour.get(r["severity_level"], (200, 200, 200)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv.save(out_path)


def run(cfg: dict, run_dir: Path, chips_dir: Path) -> Path | None:
    sev = cfg.get("severity") or {}
    if not (run_dir / "2_scores.csv").exists():
        print("[severity] 2_scores.csv 없음 — 먼저 --stage classify")
        return None

    mode = cfg["decision"].get("mode", "binary")
    farmland = (mode == "farmland")
    w_gb = sev.get("w_gb", 40.0)
    w_cf = sev.get("w_cf", 30.0)
    w_area = sev.get("w_area", 30.0)
    cf_ref = sev.get("cf_ref", 0.5)
    area_lo = math.log(max(sev.get("area_lo_m2", 1000), 1))     # 이하 전환면적은 area 항 0
    area_ref = math.log(max(sev.get("area_ref_m2", 20000), 2))  # 이상은 area 항 만점
    b_lo, b_hi = sev.get("score_bands", [30, 58])
    risk_floor = sev.get("risk_floor", 0.5)
    area_big = sev.get("area_big_m2", 30000 if not farmland else 5000)
    cf_hi = sev.get("cf_hi", 0.35)

    gbu, gb_name = _greenbelt_union(Path(sev["greenbelt"])) if sev.get("greenbelt") else (None, None)
    geoms = _parcel_geoms(Path(sev["parcels"])) if sev.get("parcels") else {}
    n_missing = 0

    scores = read_csv(run_dir / "2_scores.csv")
    print(f"[severity] {len(scores)}필지 (전량) | greenbelt={gb_name or '없음'} | "
          f"risk_floor={risk_floor} area_big={area_big}㎡ cf_hi={cf_hi}")

    rows = []
    for s in scores:
        pnu = s["pnu"]
        area = float(s.get("area_m2") or 0)
        cr = float(s.get("change_frac") or 0)
        risk = float(s["p_screen"]) if farmland else 1.0 - float(s["p_forest"])
        change_area = cr * area

        geom = geoms.get(pnu)
        inter = 0.0
        if geom is None:
            if geoms:
                n_missing += 1
        elif gbu is not None:
            inter = float(geom.intersection(gbu).area)
        gb_ratio = round(inter / geom.area, 4) if (geom is not None and geom.area) else 0.0
        in_gb = inter >= 1.0

        area_term = (math.log(max(change_area, 1.0)) - area_lo) / max(area_ref - area_lo, 1e-6)
        sc = (w_gb * (1.0 if in_gb else 0.0)
              + w_cf * min(cr / max(cf_ref, 1e-6), 1.0)
              + w_area * min(max(area_term, 0.0), 1.0))
        sc = round(min(100.0, sc), 1)
        lvl = LEVELS[2] if sc >= b_hi else (LEVELS[1] if sc >= b_lo else LEVELS[0])

        if in_gb and risk >= risk_floor:
            tier = "T1"
        elif risk >= risk_floor and (area >= area_big or cr >= cf_hi):
            tier = "T2"
        else:
            tier = "T3"

        rows.append({
            "tier": tier, "risk": round(risk, 4),
            "pnu": pnu, "jibun": s.get("jibun", ""), "jimok": s.get("jimok", ""),
            "area_m2": round(area, 1), "closest_neg": s.get("closest_neg", ""),
            "change_frac": round(cr, 4), "change_area_m2": round(change_area, 1),
            "in_greenbelt": in_gb, "gb_ratio": gb_ratio,
            "severity_score": sc, "severity_level": lvl,
            "p_forest": s.get("p_forest", ""), "p_screen": s.get("p_screen", ""),
            "chip": s.get("chip", ""),
        })

    order = {"T1": 0, "T2": 1, "T3": 2}
    rows.sort(key=lambda r: (order[r["tier"]], -r["risk"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    out = run_dir / "5_worklist.csv"
    cols = ["rank", "tier", "risk", "pnu", "jibun", "jimok", "area_m2", "closest_neg",
            "change_frac", "change_area_m2", "in_greenbelt", "gb_ratio",
            "severity_score", "severity_level", "p_forest", "p_screen", "chip"]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    tc = Counter(r["tier"] for r in rows)
    lc = Counter(r["severity_level"] for r in rows)
    summary = {
        "n": len(rows),
        "tiers": [[t, tc.get(t, 0)] for t in ("T1", "T2", "T3")],
        "levels": [[lv, lc.get(lv, 0)] for lv in reversed(LEVELS)],
        "n_in_greenbelt": sum(r["in_greenbelt"] for r in rows),
        "n_missing_geom": n_missing,
        "params": {"w_gb": w_gb, "w_cf": w_cf, "w_area": w_area, "cf_ref": cf_ref,
                   "area_lo_m2": sev.get("area_lo_m2", 1000), "area_ref_m2": sev.get("area_ref_m2", 20000),
                   "score_bands": [b_lo, b_hi],
                   "risk_floor": risk_floor, "area_big_m2": area_big, "cf_hi": cf_hi,
                   "greenbelt": gb_name, "parcels": Path(sev["parcels"]).name},
    }
    (run_dir / "5_worklist.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    _montage(rows, chips_dir, run_dir / "5_worklist.png")

    print(f"[severity] → {out}")
    print(f"[severity] 티어 T1 {tc.get('T1',0)} · T2 {tc.get('T2',0)} · T3 {tc.get('T3',0)}   "
          f"등급 높음 {lc.get('높음',0)} · 중간 {lc.get('중간',0)} · 낮음 {lc.get('낮음',0)}   "
          f"(GB {summary['n_in_greenbelt']}"
          + (f", 폴리곤미매칭 {n_missing}" if n_missing else "") + ")")
    print(f"[severity] → {run_dir / '5_worklist.png'}")
    return out
