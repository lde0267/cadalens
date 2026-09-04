#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
라벨링한 필지를 225x225 프레임에 "꽉 차게" 다시 렌더 —
작은 필지는 확대, 큰 필지는 축소해서 필지 하나가 영상에 가득 차도록.

- 대상: data/labels/imya_eval_150/labels.csv 의 PNU (손라벨한 것들)
- 창(window) = 필지 bbox 의 긴 변 * margin  (면적 하한 없음 → 필지 크기에 따라 배율이 달라짐)
- 반투명 빨강 채움 + 빨강 외곽선(선 두께는 배율에 맞춰 px 고정)
- 정사각 패딩 후 225x225 LANCZOS

산출: data/labels/imya_eval_150/previews_fit/<PNU>.jpg
      (--split 주면 라벨별 폴더에도 복사: previews_fit/<label>/<PNU>.jpg)

사용:
    python scripts/labeling/make_fit_previews.py
    python scripts/labeling/make_fit_previews.py --px 225 --margin 1.06 --split
"""
from __future__ import annotations

import argparse
import csv
import os
import sysconfig
from pathlib import Path

_SITE = sysconfig.get_paths()["purelib"]
for _v, _d in (("PROJ_LIB", "proj_data"), ("PROJ_DATA", "proj_data"), ("GDAL_DATA", "gdal_data")):
    _p = os.path.join(_SITE, "rasterio", _d)
    if os.path.isdir(_p):
        os.environ[_v] = _p

import numpy as np
import geopandas as gpd
import rasterio
from rasterio import windows
from rasterio.features import rasterize
from rasterio.transform import rowcol
from shapely.geometry import mapping
from PIL import Image, ImageDraw

DEF_PARCELS = Path("data/cadastral/derived/anseong_37714092_parcels_within.gpkg")
DEF_ORTHO = Path("data/satellite/georef/(B060)정사영상_2025_37714092_georef.tif")
DEF_SET = Path("data/labels/imya_eval_150")
RED = np.array([255, 40, 40], np.float32)


def pad_square(a: np.ndarray, fill=20) -> np.ndarray:
    h, w = a.shape[:2]
    s = max(h, w)
    o = np.full((s, s, a.shape[2]), fill, a.dtype)
    o[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
    return o


def _rings(geom):
    """폴리곤/멀티폴리곤의 모든 외곽·내곽 링 좌표열."""
    gt = geom.geom_type
    if gt == "Polygon":
        polys = [geom]
    elif gt == "MultiPolygon":
        polys = list(geom.geoms)
    else:
        return []
    out = []
    for p in polys:
        out.append(list(p.exterior.coords))
        out += [list(r.coords) for r in p.interiors]
    return out


def render_fit(ortho, geom, px: int, margin: float,
               fill_alpha: float, edge_px: float) -> Image.Image | None:
    minx, miny, maxx, maxy = geom.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    # 필지 긴 변에만 여백을 줘서 창을 잡는다 (면적 하한 없음)
    half = max(maxx - minx, maxy - miny) * margin / 2
    if half <= 0:
        return None
    win = windows.from_bounds(cx - half, cy - half, cx + half, cy + half, ortho.transform)
    win = win.round_offsets().round_lengths()
    win = win.intersection(windows.Window(0, 0, ortho.width, ortho.height))
    if win.width < 4 or win.height < 4:
        return None
    img = np.transpose(ortho.read(indexes=[1, 2, 3], window=win), (1, 2, 0)).astype(np.float32)
    wtf = ortho.window_transform(win)
    h, w = img.shape[:2]

    # 반투명 빨강 채움은 소스 해상도에서
    mask = rasterize([(mapping(geom), 1)], out_shape=(h, w), transform=wtf,
                     fill=0, all_touched=True).astype(bool)
    img[mask] = img[mask] * (1 - fill_alpha) + RED * fill_alpha
    out = np.clip(img, 0, 255).astype(np.uint8)

    # 정사각 패딩 후 최종 px 로 리사이즈
    s = max(h, w)
    off_x, off_y = (s - w) // 2, (s - h) // 2
    im = Image.fromarray(pad_square(out, fill=20)).resize((px, px), Image.LANCZOS)

    # 외곽선은 리사이즈 후 최종 픽셀 좌표에서 고정 두께로 그린다
    sc = px / s
    dr = ImageDraw.Draw(im)
    lw = max(1, round(edge_px))
    for ring in _rings(geom):
        pts = []
        for x, y in ring:
            r, c = rowcol(wtf, x, y, op=float)
            pts.append(((c + off_x) * sc, (r + off_y) * sc))
        if len(pts) >= 2:
            dr.line(pts, fill=(255, 40, 40), width=lw, joint="curve")
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parcels", type=Path, default=DEF_PARCELS)
    ap.add_argument("--ortho", type=Path, default=DEF_ORTHO)
    ap.add_argument("--set", type=Path, default=DEF_SET)
    ap.add_argument("--px", type=int, default=225)
    ap.add_argument("--margin", type=float, default=1.06,
                    help="필지 긴 변 대비 창 배율 (1.0=딱 맞음, 1.06=6%% 여백)")
    ap.add_argument("--fill-alpha", type=float, default=0.18)
    ap.add_argument("--edge-px", type=float, default=2.0)
    ap.add_argument("--split", action="store_true", help="라벨별 폴더에도 복사")
    args = ap.parse_args()

    labels_csv = args.set / "labels.csv"
    want: dict[str, str] = {}
    with open(labels_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            want[row["pnu"].strip()] = row["label"].strip()
    print(f"라벨 {len(want)}개 대상")

    gdf = gpd.read_file(args.parcels)
    gdf = gdf[gdf["PNU"].isin(want)].copy()
    print(f"gpkg 매칭 {len(gdf)}개")

    out_dir = args.set / "previews_fit"
    out_dir.mkdir(parents=True, exist_ok=True)

    done, skipped = 0, []
    with rasterio.open(args.ortho) as ortho:
        if gdf.crs and ortho.crs:
            try:
                gdf = gdf.to_crs(ortho.crs)
            except Exception:
                pass
        for r in gdf.itertuples():
            geom = r.geometry if r.geometry.is_valid else r.geometry.buffer(0)
            im = render_fit(ortho, geom, args.px, args.margin, args.fill_alpha, args.edge_px)
            if im is None:
                skipped.append(r.PNU)
                continue
            im.save(out_dir / f"{r.PNU}.jpg", quality=88)
            if args.split:
                sub = out_dir / want[r.PNU]
                sub.mkdir(exist_ok=True)
                im.save(sub / f"{r.PNU}.jpg", quality=88)
            done += 1
            if done % 25 == 0:
                print(f"  ...{done}")

    missing = sorted(set(want) - set(gdf["PNU"]))
    print(f"\n완료 {done}개 → {out_dir}")
    if skipped:
        print(f"영상 밖 skip {len(skipped)}: {skipped}")
    if missing:
        print(f"gpkg 에 없음 {len(missing)}: {missing}")


if __name__ == "__main__":
    main()
