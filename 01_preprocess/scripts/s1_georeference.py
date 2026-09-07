#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
국토지리정보원 항공사진/정사영상 TIF에 좌표계·지오레퍼런스를 주입해 GeoTIFF로 출력.

Global Mapper·Photoshop 등을 거치며 GeoTIFF 태그가 날아간 TIF는 CRS도 transform도 없다.
같이 배포된 `*메타데이터*.xml` 로 이를 복원한다. 두 가지 케이스를 지원:

(A) 항공사진 메타데이터 - <원점X좌표>/<원점Y좌표>(영상 좌상단 좌표) + <해상도>
      transform = Affine(res, 0, 원점X, 0, -res, 원점Y)
      * 주의: 원본 '항공사진(프레임)'은 정사보정이 안 돼 있어 이 방식으로는 정합되지 않는다.
        (기복변위·경사·회전). 반드시 '정사영상' 산출물을 써야 한다.

(B) 정사영상 메타데이터 - <해당도엽번호> + <지상표본거리>
      도엽번호(1:5,000)를 디코딩해 도곽 경위도를 구하고, 대상 좌표계로 투영,
      50 m 도곽 여백을 적용해 좌상단을 계산한다.
      * 원본 배포 TIF에는 georef가 들어있다. 이 경로는 그게 사라졌을 때의 '근사 복원'이며
        오차 ~1 m 수준이므로, 정밀 작업 전 지적도 기준 GCP 로 미세보정 권장.

공통:  CRS = (<투영원점>, <기준타원체>, <Easting>/<Northing>) -> EPSG (중부/GRS80/200000/600000 = EPSG:5186)

사용
----
    python 01_preprocess/scripts/s1_georeference.py             # 01_preprocess/raw/satellite/*.tif 자동
    python 01_preprocess/scripts/s1_georeference.py \
        --tif "01_preprocess/raw/satellite/(B060)정사영상_2025_37714092.tif" \
        --xml "01_preprocess/raw/satellite/(B060)정사영상메타데이터_202511001237714092.xml"

    산출: 01_preprocess/derived/*_georef.tif
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import geoenv  # noqa: E402,F401  (import 부작용: PROJ/GDAL 경로 격리 — rasterio import 전)
from common.paths import RAW_SATELLITE, DERIVED, CADASTRE_SHP  # noqa: E402

import rasterio  # noqa: E402
from rasterio.crs import CRS
from rasterio.transform import Affine

# 투영원점 이름 -> 중앙자오선(경도)
_CENTRAL_MERIDIAN = {"서부": 125.0, "중부": 127.0, "동부": 129.0, "동해": 131.0, "울릉": 131.0}

# (투영원점, false_northing) -> EPSG  [기준타원체 GRS80 = Korea 2000 계열]
#   false_northing 600000 = "...Belt 2010" 계열(현행), 500000 = 구계열
_EPSG_GRS80 = {
    ("서부", 600000): 5185, ("중부", 600000): 5186, ("동부", 600000): 5187, ("동해", 600000): 5188,
    ("서부", 500000): 5185, ("중부", 500000): 5181, ("동부", 500000): 5182, ("동해", 500000): 5183,
}

# Korea 2000 지리좌표(경위도). WGS84(4326)와 이 축척에서 차이 <1 m.
_GEOGRAPHIC_EPSG = 4737


def _text(root: ET.Element, tag: str) -> str | None:
    el = root.find(f".//{tag}")
    return el.text.strip() if el is not None and el.text else None


def _num(s: str | None):
    if not s:
        return None
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else None


# --------------------------------------------------------------------------------------
# 도엽번호 디코딩  (1:50,000 5자리  [+ 1:5,000 3자리])
# --------------------------------------------------------------------------------------
def decode_mapsheet(sheet_no: str) -> tuple[float, float, float, float]:
    """
    반환: (lon_W, lat_S, lon_E, lat_N)  [도 단위, Korea 2000 경위도]

    1:50,000 'ABCDE' : AB=위도(도), C=경도 끝자리(12C도), DE=1도x1도 블록 내
                       15'x15' 셀 1..16 (NW 기준 동->남).
    1:5,000  'FGH'   : 15'x15' 를 1.5'x1.5' 로 10x10 분할, 1..100 (NW 기준 동->남).
    """
    s = re.sub(r"\D", "", sheet_no)
    if len(s) < 5:
        raise ValueError(f"도엽번호 형식 오류: {sheet_no!r}")
    lat_deg = int(s[0:2])
    lon_deg = 120 + int(s[2])
    cell50 = int(s[3:5])
    idx = cell50 - 1
    row, col = idx // 4, idx % 4            # row: 북->남(0..3), col: 서->동(0..3)
    n50 = lat_deg + 1.0 - row * 0.25        # 블록 북단 = lat_deg+1
    w50 = lon_deg + col * 0.25

    lat_N, lat_S = n50, n50 - 0.25
    lon_W, lon_E = w50, w50 + 0.25

    rest = s[5:]
    if rest:
        n5 = int(rest[:3])
        idx5 = n5 - 1
        r5, c5 = idx5 // 10, idx5 % 10
        step = 1.5 / 60.0
        lat_N = n50 - r5 * step
        lat_S = lat_N - step
        lon_W = w50 + c5 * step
        lon_E = lon_W + step
    return lon_W, lat_S, lon_E, lat_N


# --------------------------------------------------------------------------------------
# 메타데이터 파싱
# --------------------------------------------------------------------------------------
def parse_metadata(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()

    proj = (_text(root, "투영법") or "TM").upper()
    if proj != "TM":
        raise ValueError(f"TM 외 투영법 미지원: {proj}")

    meta = dict(
        origin_name=_text(root, "투영원점") or "중부",
        ellipsoid=(_text(root, "기준타원체") or "GRS80").upper(),
        false_easting=_num(_text(root, "Easting")) or 200000.0,
        false_northing=_num(_text(root, "Northing")) or 600000.0,
        admin=_text(root, "행정구역명"),
        date=_text(root, "촬영일자") or _text(root, "촬영년도") or _text(root, "제작일자"),
        sheet_no=_text(root, "해당도엽번호") or _text(root, "해당도엽번호150000"),
        res=_num(_text(root, "해상도")) or _num(_text(root, "지상표본거리")),
    )

    ox, oy = _num(_text(root, "원점X좌표")), _num(_text(root, "원점Y좌표"))
    if ox is not None and oy is not None:
        meta["mode"], meta["ox"], meta["oy"] = "origin_xy", ox, oy
    elif meta["sheet_no"]:
        meta["mode"] = "mapsheet"
    else:
        raise ValueError("메타에 <원점X좌표>/<원점Y좌표> 도 <해당도엽번호> 도 없습니다.")

    if meta["res"] is None:
        raise ValueError("메타에 <해상도>/<지상표본거리> 가 없습니다.")
    return meta


def resolve_crs(meta: dict) -> CRS:
    key = (meta["origin_name"], int(meta["false_northing"]))
    if meta["ellipsoid"] == "GRS80" and key in _EPSG_GRS80:
        return CRS.from_epsg(_EPSG_GRS80[key])

    lon0 = _CENTRAL_MERIDIAN.get(meta["origin_name"])
    if lon0 is None:
        raise ValueError(f"알 수 없는 투영원점: {meta['origin_name']}")
    ellps = "GRS80" if meta["ellipsoid"] == "GRS80" else "bessel"
    towgs = "+towgs84=0,0,0,0,0,0,0 " if ellps == "GRS80" else ""
    proj4 = (f"+proj=tmerc +lat_0=38 +lon_0={lon0} +k=1 "
             f"+x_0={meta['false_easting']:.0f} +y_0={meta['false_northing']:.0f} "
             f"+ellps={ellps} {towgs}+units=m +no_defs")
    print(f"  ! EPSG 직접매칭 실패 -> proj4: {proj4}", file=sys.stderr)
    return CRS.from_proj4(proj4)


# --------------------------------------------------------------------------------------
# transform 계산
# --------------------------------------------------------------------------------------
def build_transform(meta: dict, crs: CRS, width: int, height: int,
                    margin: float) -> tuple[Affine, dict]:
    res = meta["res"]
    info: dict = {}

    if meta["mode"] == "origin_xy":
        ulx, uly = meta["ox"], meta["oy"]
        info["method"] = "메타 <원점X/Y> = 좌상단"
    else:
        from pyproj import Transformer
        lon_w, lat_s, lon_e, lat_n = decode_mapsheet(meta["sheet_no"])
        info["sheet"] = (lon_w, lat_s, lon_e, lat_n)
        tr = Transformer.from_crs(_GEOGRAPHIC_EPSG, crs, always_xy=True)
        corners = [tr.transform(lon, lat) for lon in (lon_w, lon_e) for lat in (lat_s, lat_n)]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        minx, maxx, maxy = min(xs), max(xs), max(ys)
        ulx, uly = minx - margin, maxy + margin
        pred_w = round((max(xs) - min(xs) + 2 * margin) / res)
        pred_h = round((maxy - min(ys) + 2 * margin) / res)
        info["method"] = f"도엽번호 {meta['sheet_no']} 디코딩 + 도곽여백 {margin:g} m (근사)"
        info["pred_size"] = (pred_w, pred_h)
        if abs(pred_w - width) > 8 or abs(pred_h - height) > 8:
            print(f"  ! 예측 픽셀크기 {pred_w}x{pred_h} vs 실제 {width}x{height} "
                  f"(차이 큼 - 도엽번호/여백 확인 필요)", file=sys.stderr)

    return Affine(res, 0.0, ulx, 0.0, -res, uly), info


# --------------------------------------------------------------------------------------
def find_pairs(sat_dir: Path) -> list[tuple[Path, Path]]:
    xmls = [x for x in sat_dir.glob("*.xml") if not x.name.lower().endswith(".aux.xml")]
    pairs = []
    for tif in sorted(sat_dir.glob("*.tif")):
        ids = re.findall(r"\d{6,}[A-Za-z0-9]*", tif.stem)
        token = ids[-1] if ids else None
        cand = [x for x in xmls if token and token in x.stem]
        match = next((x for x in cand if "메타" in x.stem), cand[0] if cand else None)
        if match is None:
            print(f"  ! 메타 XML 매칭 실패, 건너뜀: {tif.name}", file=sys.stderr)
            continue
        pairs.append((tif, match))
    return pairs


def georeference(tif: Path, xml: Path, outdir: Path, margin: float) -> None:
    meta = parse_metadata(xml)
    crs = resolve_crs(meta)
    res = meta["res"]

    outdir.mkdir(parents=True, exist_ok=True)
    gtiff_path = outdir / f"{tif.stem}_georef.tif"

    with rasterio.open(tif) as src:
        if src.crs is not None:
            print(f"  - 이미 CRS 있음({src.crs.to_string()}) - 덮어씀: {tif.name}")
        w, h = src.width, src.height
        transform, info = build_transform(meta, crs, w, h, margin)
        profile = src.profile.copy()
        profile.update(crs=crs, transform=transform, driver="GTiff",
                       tiled=True, blockxsize=512, blockysize=512,
                       compress="deflate", predictor=2, BIGTIFF="IF_SAFER")
        data = src.read()
        tags = src.tags()

    with rasterio.open(gtiff_path, "w", **profile) as dst:
        dst.write(data)
        dst.update_tags(**tags)
        dst.update_tags(AREA_OR_POINT="Area", SOURCE_TIF=tif.name,
                        SOURCE_METADATA=xml.name, CAPTURE_DATE=meta["date"] or "",
                        ADMIN=meta["admin"] or "", GEOREF_METHOD=info["method"])

    with rasterio.open(gtiff_path) as chk:
        b = chk.bounds
        print(f"  OK  {tif.name}")
        print(f"      방식       : {info['method']}")
        print(f"      CRS        : {crs.to_string()}  (EPSG:{crs.to_epsg()})")
        print(f"      size       : {chk.width} x {chk.height} px @ {res} m")
        if "sheet" in info:
            lw, ls, le, ln = info["sheet"]
            print(f"      도곽 경위도: {lw:.5f}~{le:.5f} E / {ls:.5f}~{ln:.5f} N")
            print(f"      예측 px    : {info['pred_size'][0]} x {info['pred_size'][1]}")
        print(f"      UL         : ({b.left:.2f}, {b.top:.2f})")
        print(f"      bounds     : L={b.left:.2f} B={b.bottom:.2f} R={b.right:.2f} T={b.top:.2f}")
        print(f"      일자/행정  : {meta['date']}  /  {meta['admin']}")
        print(f"      -> {gtiff_path}")
        if meta["mode"] == "mapsheet":
            print("      * 근사 복원입니다. 정밀 클리핑 전 지적도 기준 GCP 미세보정 권장.")


def maybe_check_against_cadastre(outdir: Path, cad_shp: Path) -> None:
    if not cad_shp.exists():
        return
    try:
        import geopandas as gpd
    except ImportError:
        return
    cad = gpd.read_file(cad_shp)
    cb = cad.total_bounds
    print(f"\n[검증] 지적도 bounds: L={cb[0]:.0f} B={cb[1]:.0f} R={cb[2]:.0f} T={cb[3]:.0f}")
    for gp in sorted(outdir.glob("*_georef.tif")):
        with rasterio.open(gp) as r:
            rb = r.bounds
        inside = (cb[0] <= rb.left and cb[1] <= rb.bottom
                  and cb[2] >= rb.right and cb[3] >= rb.top)
        overlap = not (rb.right < cb[0] or rb.left > cb[2]
                       or rb.top < cb[1] or rb.bottom > cb[3])
        tag = "[OK] 지적도 범위 내부" if inside else ("[OK] 겹침" if overlap else "[!!] 불일치")
        print(f"  {gp.name}: {tag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif", type=Path)
    ap.add_argument("--xml", type=Path)
    ap.add_argument("--sat-dir", type=Path, default=RAW_SATELLITE)
    ap.add_argument("--outdir", type=Path, default=DERIVED)
    ap.add_argument("--margin", type=float, default=50.0,
                    help="도엽번호 방식에서 도곽 여백(m). 기본 50 (NGII 표준).")
    ap.add_argument("--cadastre", type=Path, default=CADASTRE_SHP)
    args = ap.parse_args()

    if args.tif:
        if not args.xml:
            ap.error("--tif 지정 시 --xml 도 필요합니다.")
        pairs = [(args.tif, args.xml)]
    else:
        pairs = find_pairs(args.sat_dir)
        if not pairs:
            ap.error(f"{args.sat_dir} 에서 (tif, xml) 짝을 찾지 못했습니다.")

    print(f"처리 대상 {len(pairs)}건")
    for tif, xml in pairs:
        print(f"\n[{tif.name}]  <- {xml.name}")
        georeference(tif, xml, args.outdir, args.margin)

    maybe_check_against_cadastre(args.outdir, args.cadastre)


if __name__ == "__main__":
    main()
