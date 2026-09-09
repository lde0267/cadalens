#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
필지 폴리곤으로 정사영상을 잘라 RemoteCLIP 입력용 이미지 칩(chip)을 만든다.

- 폴리곤 내부만 남기고 바깥은 nodata(투명 + 검정)로 마스킹
- 종횡비 유지하며 정사각 패딩 후 고정 크기(기본 224x224)로 리샘플
    * RemoteCLIP / OpenCLIP(ViT-B/32·B/16·L/14) 기본 입력이 224px.
    * 저장은 RGBA PNG. 추론 시 PIL `Image.open(p).convert("RGB")` 하면
      폴리곤 바깥은 검정(0,0,0)으로 들어간다. alpha 채널을 마스크로 써도 됨.
- --no-square 를 주면 리샘플 없이 마스킹된 bbox 크롭 원본 크기로 저장

기본 입력: 01_preprocess/derived/anseong_37714092_parcels_within.gpkg   (s2 산출)
           01_preprocess/derived/(B060)정사영상_2025_37714092_georef.tif  (s1 산출)
산출:     01_preprocess/chips/<sheet>_<mode>/           (--jimok 단일)
          01_preprocess/chips/<sheet>_<group>_<mode>/   (--group 또는 --jimok 콤마목록)
          (index.csv + <pnu>_<jibun>.png + _montage.png)

예시:
    # 필지별 전체 칩 — 임야 / 농지(전·답·과) 분리, 해당 지목 전 필지
    python 01_preprocess/scripts/s3_make_chips.py --group imya   --mode real --limit 0
    python 01_preprocess/scripts/s3_make_chips.py --group nongji --mode mask --limit 0

    python 01_preprocess/scripts/s3_make_chips.py --jimok 임 --mode mask --limit 0
    python 01_preprocess/scripts/s3_make_chips.py --jimok 전,답,과 --mode real --limit 0
    python 01_preprocess/scripts/s3_make_chips.py --jimok all --mode context --limit 0
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import geoenv  # noqa: E402,F401  (import 부작용: PROJ/GDAL 경로 격리)
from common.paths import PARCELS as _PARCELS, ORTHO_GEOREF as _ORTHO, chip_dir  # noqa: E402

import cv2  # noqa: E402  배열 리샘플 전용 (파일 I/O 에는 쓰지 않음: 한글 경로 문제)
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import geopandas as gpd  # noqa: E402
import rasterio  # noqa: E402
from rasterio import windows  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from shapely.geometry import mapping  # noqa: E402

DEF_PARCELS = _PARCELS
DEF_ORTHO = _ORTHO
# --outdir 미지정 시 mode 별로 01_preprocess/chips/<sheet>_<mode>/ 로 자동 결정

# 필지 그룹 → 지목부호 집합 (필지별 칩을 지목군 단위로 분리 생성)
GROUPS = {
    "imya": ["임"],            # 임야
    "nongji": ["전", "답", "과"],  # 농지 = 전 · 답 · 과수원
}


def sanitize(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "_", str(s)).strip("_")


def pad_to_square(arr: np.ndarray, fill: int) -> np.ndarray:
    """(H,W,C) -> (S,S,C), S=max(H,W), 가운데 정렬, 나머지 fill."""
    h, w = arr.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    out = np.full((s, s, arr.shape[2]), fill, dtype=arr.dtype)
    out[top:top + h, left:left + w] = arr
    return out


def make_chip_real(ortho: rasterio.DatasetReader, geom, size: int,
                   pad_m: float, max_aspect: float):
    """
    mode='real' : 필지 bbox 를 224 프레임에 꽉 채운다.
      - 창 = 필지 bounding box + pad_m 여백 (정사각화 X)
      - 종횡비가 max_aspect 초과면 짧은쪽을 실영상으로 대칭 확장해 cap
      - 바깥(폴리곤 밖)도 원본 정사영상 픽셀 그대로 (검정 채움 X, 외곽선 X)
      - size x size 로 스트레치 리사이즈
      - alpha = 폴리곤 마스크: 내부 255, 외부 0
    반환: (chip[h,w,4], valid_ratio, native_hw, eff_gsd_m)
    """
    minx, miny, maxx, maxy = geom.bounds
    minx, miny, maxx, maxy = minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m
    bw, bh = maxx - minx, maxy - miny
    if max(bw, bh) / max(min(bw, bh), 1e-6) > max_aspect:      # 종횡비 cap
        if bw > bh:
            g = (bw / max_aspect - bh) / 2
            miny, maxy = miny - g, maxy + g
        else:
            g = (bh / max_aspect - bw) / 2
            minx, maxx = minx - g, maxx + g

    win = windows.from_bounds(minx, miny, maxx, maxy, ortho.transform)
    win = win.round_offsets().round_lengths().intersection(
        windows.Window(0, 0, ortho.width, ortho.height))
    if win.width < 2 or win.height < 2:
        return None
    img = np.transpose(ortho.read(indexes=[1, 2, 3], window=win), (1, 2, 0))
    wtf = ortho.window_transform(win)
    mask = rasterize([(mapping(geom), 1)], out_shape=img.shape[:2], transform=wtf,
                     fill=0, all_touched=True).astype(np.uint8)
    valid_ratio = float(mask.mean())
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if max(h, w) > size else cv2.INTER_CUBIC
    rgb_r = cv2.resize(img, (size, size), interpolation=interp)
    a_r = cv2.resize(mask * 255, (size, size), interpolation=cv2.INTER_NEAREST)
    return np.dstack([rgb_r, a_r]), valid_ratio, (h, w), round(ortho.res[0] * max(h, w) / size, 4)


def make_chip(ortho: rasterio.DatasetReader, geom, size: int, pad_m: float,
              square: bool, fill: int, mode: str, min_window_m: float,
              context_frac: float, outline: bool):
    """
    mode='mask'    : 폴리곤 내부만 남기고 바깥 nodata (RGBA, alpha=마스크)
    mode='context' : 마스킹 없이 필지 중심 정사각 창을 실장면 그대로 (RGB[A]),
                     min_window_m/context_frac 로 창 크기 확보, outline=폴리곤 외곽선
    (mode='real' 은 make_chip_real 에서 처리)
    반환: (chip[h,w,4], valid_ratio, native_hw, eff_gsd_m)
    """
    minx, miny, maxx, maxy = geom.bounds

    if mode == "context":
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        side = max(maxx - minx, maxy - miny, min_window_m) * (1 + context_frac) + 2 * pad_m
        minx, maxx = cx - side / 2, cx + side / 2
        miny, maxy = cy - side / 2, cy + side / 2
    elif pad_m:
        minx, miny, maxx, maxy = minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m

    win = windows.from_bounds(minx, miny, maxx, maxy, ortho.transform)
    win = win.round_offsets().round_lengths()
    win = win.intersection(windows.Window(0, 0, ortho.width, ortho.height))
    if win.width < 1 or win.height < 1:
        return None

    img = np.transpose(ortho.read(indexes=[1, 2, 3], window=win), (1, 2, 0))
    wtf = ortho.window_transform(win)
    mask = rasterize([(mapping(geom), 1)], out_shape=img.shape[:2],
                     transform=wtf, fill=0, all_touched=True).astype(bool)
    valid_ratio = float(mask.mean())

    if mode == "context":
        rgb = img.copy()
        if outline:
            edge = rasterize([(mapping(geom.boundary.buffer(ortho.res[0] * 1.2)), 1)],
                             out_shape=img.shape[:2], transform=wtf, fill=0,
                             all_touched=True).astype(bool)
            rgb[edge] = (255, 30, 30)
        alpha = np.full(img.shape[:2], 255, np.uint8)
    else:
        rgb = img.copy()
        rgb[~mask] = fill
        alpha = np.where(mask, 255, 0).astype(np.uint8)

    rgba = np.dstack([rgb, alpha])
    native_hw = rgba.shape[:2]

    if square:
        rgba = pad_to_square(rgba, fill)
        interp = cv2.INTER_AREA if rgba.shape[0] > size else cv2.INTER_CUBIC
        rgb_r = cv2.resize(rgba[..., :3], (size, size), interpolation=interp)
        a_r = cv2.resize(rgba[..., 3], (size, size), interpolation=cv2.INTER_NEAREST)
        rgba = np.dstack([rgb_r, a_r])

    eff_gsd = ortho.res[0] * max(native_hw) / size if square else ortho.res[0]
    return rgba, valid_ratio, native_hw, round(eff_gsd, 4)


def montage(paths, cols, cell, out_path):
    rows = (len(paths) + cols - 1) // cols
    canvas = np.full((rows * cell, cols * cell, 3), 32, np.uint8)
    for i, p in enumerate(paths):
        im = np.asarray(Image.open(p).convert("RGBA"))
        a = im[..., 3:4] / 255.0                                # alpha 를 진회색 위에 합성
        im = (im[..., :3] * a + 32 * (1 - a)).astype(np.uint8)
        im = np.asarray(Image.fromarray(im).resize((cell, cell), Image.BILINEAR))
        r, c = divmod(i, cols)
        canvas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = im
    Image.fromarray(canvas).save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parcels", type=Path, default=DEF_PARCELS)
    ap.add_argument("--ortho", type=Path, default=DEF_ORTHO)
    ap.add_argument("--outdir", type=Path, default=None,
                    help="미지정 시 01_preprocess/chips/<sheet>_<mode>/ 자동")
    ap.add_argument("--size", type=int, default=224, help="칩 한 변 픽셀 (RemoteCLIP=224)")
    ap.add_argument("--limit", type=int, default=10, help="0 이면 전체")
    ap.add_argument("--group", choices=sorted(GROUPS),
                    help="필지 그룹으로 지목 지정: imya=임 / nongji=전,답,과. "
                         "지정 시 --jimok 무시, 산출 폴더는 <sheet>_<group>_<mode>/")
    ap.add_argument("--jimok", default="임",
                    help="지목부호 필터. 단일(임) · 콤마목록(전,답,과) · 'all'(전체). "
                         "--group 을 주면 무시됨")
    ap.add_argument("--select", choices=["area", "random", "head"], default="area",
                    help="area=면적 큰 순, random=무작위, head=파일 순서")
    ap.add_argument("--pad-m", type=float, default=0.0, help="창 바깥 여유(m)")
    ap.add_argument("--mode", choices=["mask", "context", "real"], default="mask",
                    help="mask=폴리곤만 nodata / context=실장면+외곽선 / "
                         "real=필지 bbox를 224에 꽉 채움, 밖도 실영상, alpha=폴리곤(1/0)")
    ap.add_argument("--real-pad-m", type=float, default=1.5,
                    help="real 모드 bbox 여백(m)")
    ap.add_argument("--real-max-aspect", type=float, default=4.0,
                    help="real 모드 최대 종횡비 (초과분은 짧은쪽을 실영상으로 확장)")
    ap.add_argument("--min-window-m", type=float, default=80.0,
                    help="context 모드 최소 정사각 창 한 변(m)")
    ap.add_argument("--context-frac", type=float, default=0.15,
                    help="context 모드 필지 대비 여백 비율")
    ap.add_argument("--no-outline", dest="outline", action="store_false",
                    help="context 모드에서 폴리곤 외곽선 표시 안 함")
    ap.add_argument("--no-square", dest="square", action="store_false",
                    help="정사각 리샘플 없이 원본 크롭으로 저장")
    ap.add_argument("--fill", type=int, default=0, help="nodata/패딩 채움값 (기본 0=검정)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # 지목 필터 결정: --group(imya/nongji) 우선 → --jimok(단일 · 콤마목록 · all)
    if args.group:
        jimoks = list(GROUPS[args.group])
        filter_tag = args.group
        jimok_desc = f"{args.group}({','.join(jimoks)})"
    elif args.jimok.lower() == "all":
        jimoks, filter_tag, jimok_desc = None, None, "all"
    else:
        jimoks = [j.strip() for j in args.jimok.split(",") if j.strip()]
        filter_tag = "".join(jimoks) if len(jimoks) > 1 else None
        jimok_desc = ",".join(jimoks)

    if args.outdir is None:
        args.outdir = chip_dir(args.mode, group=filter_tag)

    gdf = gpd.read_file(args.parcels)
    if "JIMOK" not in gdf.columns:
        raise SystemExit("입력에 JIMOK 컬럼이 없습니다. extract_parcels_within_image.py 산출물을 쓰세요.")
    if jimoks is not None:
        gdf = gdf[gdf["JIMOK"].isin(jimoks)]
    if gdf.empty:
        raise SystemExit(f"지목 '{jimok_desc}' 필지가 없습니다.")

    if "area_m2" not in gdf.columns:
        gdf["area_m2"] = gdf.geometry.area
    if args.select == "area":
        gdf = gdf.sort_values("area_m2", ascending=False)
    elif args.select == "random":
        gdf = gdf.sample(frac=1.0, random_state=args.seed)
    if args.limit and args.limit > 0:
        gdf = gdf.head(args.limit)
    gdf = gdf.reset_index(drop=True)

    with rasterio.open(args.ortho) as ortho:
        if gdf.crs and ortho.crs and gdf.crs.to_authority() != ortho.crs.to_authority():
            gdf = gdf.to_crs(ortho.crs)

        args.outdir.mkdir(parents=True, exist_ok=True)
        rows, saved = [], []
        print(f"정사영상: {args.ortho.name}  ({ortho.res[0]} m/px)")
        print(f"대상 필지: 지목='{jimok_desc}', {args.select} 상위 {len(gdf)}개, "
              f"칩 {'%dx%d' % (args.size, args.size) if args.square else '원본크롭'}\n")

        for r in gdf.itertuples():
            geom = r.geometry
            if geom is None or geom.is_empty:
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
            if args.mode == "real":
                res = make_chip_real(ortho, geom, args.size, args.real_pad_m,
                                     args.real_max_aspect)
            else:
                res = make_chip(ortho, geom, args.size, args.pad_m, args.square, args.fill,
                                args.mode, args.min_window_m, args.context_frac, args.outline)
            if res is None:
                print(f"  skip {r.PNU} (영상 범위 밖)")
                continue
            rgba, vr, nhw, gsd = res
            jibun = getattr(r, "JIBUN", "")
            name = f"{r.PNU}_{sanitize(jibun)}.png"
            path = args.outdir / name
            Image.fromarray(rgba, "RGBA").save(path)
            saved.append(path)
            c = geom.centroid
            rows.append(dict(pnu=r.PNU, jibun=jibun, jimok=r.JIMOK,
                             area_m2=round(float(r.area_m2), 2),
                             cx=round(c.x, 3), cy=round(c.y, 3),
                             native_h=nhw[0], native_w=nhw[1],
                             gsd_m=gsd, valid_ratio=round(vr, 4),
                             size=args.size if args.square else "",
                             chip=name))
            if len(gdf) <= 40 or len(rows) % 200 == 0:
                print(f"  OK  {len(rows):>5}/{len(gdf)}  {name}  | {r.area_m2:>9.1f} m² "
                      f"| 원본 {nhw[1]}x{nhw[0]}px | 유효 {vr:5.1%}")

        if not rows:
            raise SystemExit("생성된 칩이 없습니다.")

        import csv
        idx = args.outdir / "index.csv"
        with open(idx, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        mont = args.outdir / "_montage.png"
        montage(saved[:100], cols=10, cell=args.size if args.square else 224, out_path=mont)

    print(f"\n칩 {len(rows)}개 → {args.outdir}")
    print(f"인덱스   → {idx}")
    print(f"미리보기 → {mont}")


if __name__ == "__main__":
    main()
