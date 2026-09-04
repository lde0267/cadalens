#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
지적도(연속지적도 필지 폴리곤) 중 항공영상 범위에 '완전히 포함'되는 필지만 추출.

절차
----
1. 지적도 SHP 와 항공영상의 좌표계를 비교/확인.
2. 항공영상 footprint(외곽 사각형)를 구한다.
     - georef GeoTIFF 가 있으면 그 bounds 사용
     - 없으면 메타데이터 XML(원점좌표 + 해상도 + 영상 픽셀크기)로 계산
3. 지적도를 영상 CRS 로 (필요시) 재투영.
4. bbox 로 1차 필터 후, geom.within(footprint) 로 '완전 포함' 필지만 선별.
5. GeoPackage 로 저장 (+ 통계 출력).

사용
----
    python scripts/extract_parcels_within_image.py
    python scripts/extract_parcels_within_image.py --margin 2.0   # 경계 2m 안쪽으로 축소
"""
from __future__ import annotations

import argparse
import os
import re
import sysconfig
import xml.etree.ElementTree as ET
from pathlib import Path

# --- PROJ/GDAL 데이터 경로 격리 (georeference_from_metadata.py 와 동일 이유) ----
_SITE = sysconfig.get_paths()["purelib"]
_RIO_PROJ = os.path.join(_SITE, "rasterio", "proj_data")
_RIO_GDAL = os.path.join(_SITE, "rasterio", "gdal_data")
if os.path.isdir(_RIO_PROJ):
    os.environ["PROJ_LIB"] = _RIO_PROJ
    os.environ["PROJ_DATA"] = _RIO_PROJ
if os.path.isdir(_RIO_GDAL):
    os.environ["GDAL_DATA"] = _RIO_GDAL
# -----------------------------------------------------------------------------

import geopandas as gpd
import rasterio
from shapely.geometry import box

DEF_CAD = Path("data/cadastral/LSMD_CONT_LDREG_경기_안성시/LSMD_CONT_LDREG_41550_202608.shp")
DEF_GEOREF = Path("data/satellite/georef/(B060)정사영상_2025_37714092_georef.tif")
DEF_RAWTIF = Path("data/satellite/(B060)정사영상_2025_37714092.tif")
DEF_XML = Path("data/satellite/(B060)정사영상메타데이터_202511001237714092.xml")
DEF_OUT = Path("data/cadastral/derived/anseong_37714092_parcels_within.gpkg")

TARGET_EPSG = 5186  # Korea 2000 / Central Belt 2010


def footprint_from_georef(path: Path):
    with rasterio.open(path) as ds:
        b = ds.bounds
        crs = ds.crs
    return box(b.left, b.bottom, b.right, b.top), crs, f"georef GeoTIFF {path.name}"


def footprint_from_metadata(xml_path: Path, raw_tif: Path):
    root = ET.parse(xml_path).getroot()

    def t(tag):
        el = root.find(f".//{tag}")
        return el.text.strip() if el is not None and el.text else None

    ox, oy = float(t("원점X좌표")), float(t("원점Y좌표"))
    res = float(re.search(r"[-+]?\d*\.?\d+", t("해상도")).group())
    with rasterio.open(raw_tif) as ds:
        w, h = ds.width, ds.height
    left, top = ox, oy
    right, bottom = ox + w * res, oy - h * res
    from rasterio.crs import CRS
    return (box(left, bottom, right, top),
            CRS.from_epsg(TARGET_EPSG),
            f"metadata XML (원점 {ox:.0f},{oy:.0f} · {res} m · {w}x{h} px)")


def guess_jimok(jibun: str) -> str:
    """지번 문자열 끝의 한글(지목부호)만 추출. 예: '24-5대' -> '대', '산12임' -> '임'."""
    if not jibun:
        return ""
    m = re.search(r"([가-힣]+)\s*$", jibun.strip())
    return m.group(1) if m else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cadastre", type=Path, default=DEF_CAD)
    ap.add_argument("--georef", type=Path, default=DEF_GEOREF)
    ap.add_argument("--raw-tif", type=Path, default=DEF_RAWTIF)
    ap.add_argument("--xml", type=Path, default=DEF_XML)
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    ap.add_argument("--layer", default="parcels_within_image")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="footprint 를 안쪽으로 줄일 여유(m). 가장자리 픽셀 오염 방지용.")
    ap.add_argument("--predicate", default="within", choices=["within", "covered_by"],
                    help="within=경계 접촉도 제외(엄격), covered_by=경계 접촉 허용")
    args = ap.parse_args()

    # 1) footprint + 영상 CRS
    if args.georef.exists():
        footprint, img_crs, src_desc = footprint_from_georef(args.georef)
    else:
        footprint, img_crs, src_desc = footprint_from_metadata(args.xml, args.raw_tif)
    if args.margin:
        footprint = footprint.buffer(-args.margin)

    # 2) 지적도 로드 & 좌표계 확인
    cad = gpd.read_file(args.cadastre)
    print("=" * 78)
    print("좌표계 확인")
    print("-" * 78)
    print(f"  지적도 SHP  : {cad.crs.to_string() if cad.crs else 'None'}")
    print(f"                EPSG={cad.crs.to_epsg()}  "
          f"proj4={cad.crs.to_proj4() if cad.crs else '-'}")
    print(f"  항공영상    : {img_crs.to_string()}  EPSG={img_crs.to_epsg()}   ({src_desc})")
    same = (cad.crs is not None) and (
        cad.crs.to_epsg() == img_crs.to_epsg()
        or cad.crs.to_authority() == img_crs.to_authority()
    )
    # EPSG 코드가 없어도 투영 파라미터가 같은지 최종 확인
    if not same and cad.crs is not None:
        a, b = cad.crs.to_dict(), img_crs.to_dict()
        keys = ("proj", "lat_0", "lon_0", "k", "x_0", "y_0", "ellps", "datum")
        same = all(a.get(k) == b.get(k) for k in keys)
    print(f"  --> {'동일 좌표계 (재투영 불필요)' if same else '좌표계 상이 → 영상 CRS 로 재투영'}")
    print("=" * 78)

    if cad.crs is None:
        raise SystemExit("지적도에 좌표계가 없습니다(.prj 확인 필요).")
    cad_img = cad.to_crs(img_crs) if not same else cad

    # 3) footprint 범위 정보
    fb = footprint.bounds
    print(f"영상 footprint bounds: L={fb[0]:.2f} B={fb[1]:.2f} R={fb[2]:.2f} T={fb[3]:.2f}"
          f"  (margin={args.margin} m)")
    print(f"지적도 전체 필지 수  : {len(cad_img):,}")

    # 4) bbox 1차 필터 → 정밀 포함 판정
    cand = cad_img.cx[fb[0]:fb[2], fb[1]:fb[3]]
    print(f"bbox 교차 후보       : {len(cand):,}")

    pred = cand.geometry.covered_by(footprint) if args.predicate == "covered_by" \
        else cand.geometry.within(footprint)
    inside = cand[pred].copy()
    print(f"'{args.predicate}' 완전 포함 : {len(inside):,}")

    if inside.empty:
        raise SystemExit("완전히 포함되는 필지가 없습니다.")

    # 5) 지오메트리 정리 + 속성 보강 + 저장
    n_bad = int((~inside.geometry.is_valid).sum())
    if n_bad:
        inside["geometry"] = inside.geometry.make_valid()
        inside = inside[inside.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        print(f"invalid 지오메트리 {n_bad}건 → make_valid 로 보정")

    inside["JIMOK"] = inside["JIBUN"].map(guess_jimok) if "JIBUN" in inside.columns else ""
    inside["area_m2"] = inside.geometry.area.round(3)

    print("\n지목부호별 분포(추정):")
    print(inside["JIMOK"].value_counts().to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    inside.to_file(args.out, layer=args.layer, driver="GPKG")
    print(f"\n저장 완료: {args.out}  (layer: {args.layer}, {len(inside):,} 필지)")


if __name__ == "__main__":
    main()
