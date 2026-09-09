#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
임야 손라벨 세트 생성 — 정사영상 위에 필지를 빨갛게 칠한 미리보기 + 자체완결 라벨링 HTML.

- data/cadastral/derived/anseong_37714092_parcels_within.gpkg 에서 지목 '임' 필지
- 면적 5분위 층화표집으로 N개(기본 150)
- 각 필지: (a) 맥락 뷰(넓게)  (b) 근접 뷰(bbox 타이트)  둘 다 폴리곤 반투명 빨강 + 외곽선
- label_tool.html : 이미지 base64 내장, 키보드(1 임야 / 2 형질변경 / 3 보류 / 4 제외),
                    localStorage 자동저장, "결과 저장(labels.json)" 다운로드 버튼
    * 제외 = 칩/영상 자체가 이상(구름·엉뚱한 필지·심한 왜곡 등) → 라벨로 쓰지 않음.
      finalize_labels.py 가 labels.csv 에서 빼고 excluded.csv 로 따로 기록.

산출: data/labels/imya_eval_150/
        previews/*.jpg   manifest.csv   label_tool.html

사용:
    python scripts/labeling/make_label_set.py
    python scripts/labeling/make_label_set.py --n 150 --seed 42
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common import geoenv  # noqa: E402,F401  (import 부작용: PROJ/GDAL 경로 격리)
from common.paths import PARCELS, ORTHO_GEOREF, LABEL_SET  # noqa: E402

import numpy as np  # noqa: E402
import geopandas as gpd  # noqa: E402
import rasterio  # noqa: E402
from rasterio import windows  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from shapely.geometry import mapping  # noqa: E402
from PIL import Image  # noqa: E402

DEF_PARCELS = PARCELS
DEF_ORTHO = ORTHO_GEOREF
DEF_OUT = LABEL_SET
JIMOK = "임"
RED = np.array([255, 40, 40], np.float32)


def pad_square(a: np.ndarray, fill=0) -> np.ndarray:
    h, w = a.shape[:2]
    s = max(h, w)
    o = np.full((s, s, a.shape[2]), fill, a.dtype)
    o[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
    return o


def render(ortho, geom, margin_factor: float, min_side_m: float,
           px: int, fill_alpha: float, edge_m: float) -> Image.Image:
    minx, miny, maxx, maxy = geom.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half = max(max(maxx - minx, maxy - miny) * margin_factor, min_side_m) / 2
    win = windows.from_bounds(cx - half, cy - half, cx + half, cy + half, ortho.transform)
    win = win.round_offsets().round_lengths()
    win = win.intersection(windows.Window(0, 0, ortho.width, ortho.height))
    if win.width < 4 or win.height < 4:
        return None
    img = np.transpose(ortho.read(indexes=[1, 2, 3], window=win), (1, 2, 0)).astype(np.float32)
    wtf = ortho.window_transform(win)
    hw = img.shape[:2]
    mask = rasterize([(mapping(geom), 1)], out_shape=hw, transform=wtf,
                     fill=0, all_touched=True).astype(bool)
    edge = rasterize([(mapping(geom.boundary.buffer(edge_m)), 1)], out_shape=hw,
                     transform=wtf, fill=0, all_touched=True).astype(bool)
    img[mask] = img[mask] * (1 - fill_alpha) + RED * fill_alpha
    img[edge] = RED
    out = np.clip(img, 0, 255).astype(np.uint8)
    out = pad_square(out, fill=20)
    im = Image.fromarray(out).resize((px, px), Image.LANCZOS)
    return im


def datauri(im: Image.Image, q=78) -> str:
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def sample_stratified(gdf: gpd.GeoDataFrame, n: int, seed: int) -> gpd.GeoDataFrame:
    g = gdf.copy()
    g["_a"] = g.geometry.area
    g = g.sort_values("_a").reset_index(drop=True)
    bins = np.array_split(g.index.values, 5)
    per = n // 5
    rng = np.random.default_rng(seed)
    picks = []
    for b in bins:
        take = min(per, len(b))
        picks += list(rng.choice(b, size=take, replace=False))
    # 부족분은 남은 데서 무작위
    rem = [i for i in g.index.values if i not in set(picks)]
    while len(picks) < n and rem:
        j = int(rng.integers(len(rem)))
        picks.append(rem.pop(j))
    return g.loc[sorted(picks)].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parcels", type=Path, default=DEF_PARCELS)
    ap.add_argument("--ortho", type=Path, default=DEF_ORTHO)
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jimok", default=JIMOK)
    ap.add_argument("--ctx-px", type=int, default=520)
    ap.add_argument("--tight-px", type=int, default=340)
    args = ap.parse_args()

    gdf = gpd.read_file(args.parcels)
    gdf = gdf[gdf["JIMOK"] == args.jimok].copy()
    if gdf.empty:
        raise SystemExit("임야 필지 없음")
    sel = sample_stratified(gdf, args.n, args.seed)
    print(f"임야 {len(gdf)} 중 {len(sel)}개 층화표집 (seed {args.seed})")

    prev = args.out / "previews"
    prev.mkdir(parents=True, exist_ok=True)

    items, rows = [], []
    with rasterio.open(args.ortho) as ortho:
        if sel.crs and ortho.crs and sel.crs.to_authority() != (ortho.crs.to_authority() if ortho.crs else None):
            try:
                sel = sel.to_crs(ortho.crs)
            except Exception:
                pass
        for k, r in enumerate(sel.itertuples()):
            geom = r.geometry if r.geometry.is_valid else r.geometry.buffer(0)
            ctx = render(ortho, geom, 3.0, 130.0, args.ctx_px, 0.28, 0.5)
            tight = render(ortho, geom, 1.25, 22.0, args.tight_px, 0.22, 0.35)
            if ctx is None or tight is None:
                print(f"  skip {r.PNU} (영상 밖)")
                continue
            ctx.save(prev / f"{r.PNU}_ctx.jpg", quality=82)
            tight.save(prev / f"{r.PNU}_tight.jpg", quality=82)
            area = float(getattr(r, "area_m2", geom.area))
            items.append(dict(pnu=r.PNU, jibun=r.JIBUN, area=round(area, 1),
                              ctx=datauri(ctx), tight=datauri(tight)))
            rows.append(dict(idx=k, pnu=r.PNU, jibun=r.JIBUN, area_m2=round(area, 1)))
            if (k + 1) % 25 == 0:
                print(f"  ...{k + 1}/{len(sel)}")

    with open(args.out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    html = build_html(items)
    (args.out / "label_tool.html").write_text(html, encoding="utf-8")
    mb = len((args.out / "label_tool.html").read_bytes()) / 1e6
    print(f"\n미리보기 {len(items)}개 → {prev}")
    print(f"manifest → {args.out / 'manifest.csv'}")
    print(f"라벨링 도구 → {args.out / 'label_tool.html'}  ({mb:.1f} MB)")
    print(f"\n브라우저로 label_tool.html 열고 1/2/3 키로 라벨 → '결과 저장' 버튼 → labels.json")
    print(f"그다음:  python scripts/labeling/finalize_labels.py <labels.json 경로>")


def build_html(items: list[dict]) -> str:
    data = json.dumps(items, ensure_ascii=False)
    return r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>임야 손라벨</title><style>
*{box-sizing:border-box}body{margin:0;font:14px/1.5 -apple-system,'Malgun Gothic',sans-serif;
background:#111;color:#eee}
header{position:sticky;top:0;background:#1b1b1b;border-bottom:1px solid #333;padding:10px 16px;
display:flex;gap:18px;align-items:center;flex-wrap:wrap;z-index:5}
.bar{flex:1;min-width:160px;height:10px;background:#333;border-radius:5px;overflow:hidden}
.bar>i{display:block;height:100%;background:#4caf50}
.cnt span{display:inline-block;margin-left:10px;padding:2px 8px;border-radius:10px;background:#2a2a2a}
button{font:inherit;padding:8px 14px;border:1px solid #444;background:#2a2a2a;color:#eee;
border-radius:6px;cursor:pointer}button:hover{background:#383838}
#wrap{max-width:1180px;margin:0 auto;padding:16px}
.imgs{display:flex;gap:14px;flex-wrap:wrap;justify-content:center}
.imgs figure{margin:0}.imgs img{max-width:100%;border:1px solid #333;border-radius:6px;background:#000;display:block}
figcaption{text-align:center;color:#999;font-size:12px;margin-top:4px}
.meta{text-align:center;margin:12px 0;color:#bbb}
.acts{display:flex;gap:12px;justify-content:center;margin:16px 0}
.acts .b{font-size:16px;padding:12px 26px}
.imya{border-color:#4caf50;color:#8bd68b}.chg{border-color:#e05555;color:#ef9a9a}.hold{border-color:#888}
.excl{border-color:#c9a227;color:#e0c975}
.nav{display:flex;gap:10px;justify-content:center;margin-top:8px}
.tag{font-weight:700}.tag.imya{color:#8bd68b}.tag.chg{color:#ef9a9a}.tag.hold{color:#bbb}.tag.excl{color:#e0c975}
kbd{background:#333;border-radius:4px;padding:1px 6px;border:1px solid #555;font-size:11px}
.done{opacity:.5}
</style></head><body>
<header>
  <strong>임야 손라벨</strong>
  <div class="bar"><i id="pi"></i></div>
  <div class="cnt" id="cnt"></div>
  <button id="exp">결과 저장 (labels.json)</button>
  <button id="jumpNext">미분류로 ▶</button>
</header>
<div id="wrap">
  <div class="meta" id="meta"></div>
  <div class="imgs">
    <figure><img id="imgC"><figcaption>맥락 (넓게)</figcaption></figure>
    <figure><img id="imgT"><figcaption>근접 (필지)</figcaption></figure>
  </div>
  <div class="acts">
    <button class="b imya" data-l="임야">임야 <kbd>1</kbd></button>
    <button class="b chg" data-l="형질변경">형질변경 <kbd>2</kbd></button>
    <button class="b hold" data-l="보류">보류 <kbd>3</kbd></button>
    <button class="b excl" data-l="제외">제외 <kbd>4</kbd></button>
  </div>
  <div class="nav">
    <button id="prev">◀ 이전 <kbd>←</kbd></button>
    <span id="pos" style="align-self:center"></span>
    <button id="next">다음 ▶ <kbd>→</kbd></button>
  </div>
  <p style="text-align:center;color:#888;font-size:12px;margin-top:20px">
    빨간 영역이 판정 대상 필지. 나무 수관이 덮여있으면 <b>임야</b>,
    벌채·나지·조성·건물·주차/야적·태양광·묘지·관통도로 등이면 <b>형질변경</b>,
    판단 어려우면 <b>보류</b>. 칩/영상 자체가 이상(구름·엉뚱한 필지·심한 왜곡)이면
    <b>제외</b> — 평가 라벨로 쓰지 않음. 라벨은 브라우저에 자동저장됨.
  </p>
</div>
<script>
const DATA = __DATA__;
const KEY = 'imya_eval_labels_v1';
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch(e){}
let i = 0;
// 첫 미분류로 시작
for (let k=0;k<DATA.length;k++){ if(!labels[DATA[k].pnu]){ i=k; break; } }

const $ = s => document.querySelector(s);
function save(){ localStorage.setItem(KEY, JSON.stringify(labels)); }
function render(){
  const d = DATA[i];
  $('#imgC').src = d.ctx; $('#imgT').src = d.tight;
  const cur = labels[d.pnu];
  const tagCls = cur ? ({'임야':'imya','형질변경':'chg','보류':'hold','제외':'excl'}[cur.label]) : '';
  $('#meta').innerHTML = `<b>${d.jibun}</b> · ${d.area.toLocaleString()} m² · ${d.pnu}` +
     (cur ? ` &nbsp;→ <span class="tag ${tagCls}">${cur.label}</span>` : '');
  $('#pos').textContent = `${i+1} / ${DATA.length}`;
  const n = Object.keys(labels).length;
  $('#pi').style.width = (100*n/DATA.length) + '%';
  let a=0,b=0,c=0,d=0; for(const k in labels){const l=labels[k].label;
    if(l==='임야')a++;else if(l==='형질변경')b++;else if(l==='제외')d++;else c++;}
  $('#cnt').innerHTML = `완료 <b>${n}</b>/${DATA.length}` +
     `<span class="tag imya">임야 ${a}</span><span class="tag chg">형질변경 ${b}</span>` +
     `<span class="tag hold">보류 ${c}</span><span class="tag excl">제외 ${d}</span>`;
}
function setLabel(l){
  const d = DATA[i];
  labels[d.pnu] = {pnu:d.pnu, jibun:d.jibun, area_m2:d.area, label:l, ts:new Date().toISOString()};
  save();
  // 다음 미분류로
  let j=i+1; while(j<DATA.length && labels[DATA[j].pnu]) j++;
  i = j<DATA.length ? j : Math.min(i+1, DATA.length-1);
  render();
}
document.querySelectorAll('.acts .b').forEach(btn =>
  btn.addEventListener('click', () => setLabel(btn.dataset.l)));
$('#prev').onclick = () => { i=Math.max(0,i-1); render(); };
$('#next').onclick = () => { i=Math.min(DATA.length-1,i+1); render(); };
$('#jumpNext').onclick = () => { let j=0; for(;j<DATA.length;j++) if(!labels[DATA[j].pnu]){i=j;break;} render(); };
$('#exp').onclick = () => {
  const arr = DATA.filter(d=>labels[d.pnu]).map(d=>labels[d.pnu]);
  const blob = new Blob([JSON.stringify(arr,null,1)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'labels.json'; a.click();
};
window.addEventListener('keydown', e => {
  if(e.key==='1') setLabel('임야');
  else if(e.key==='2') setLabel('형질변경');
  else if(e.key==='3') setLabel('보류');
  else if(e.key==='4') setLabel('제외');
  else if(e.key==='ArrowLeft'){ i=Math.max(0,i-1); render(); }
  else if(e.key==='ArrowRight'){ i=Math.min(DATA.length-1,i+1); render(); }
});
render();
</script></body></html>""".replace("__DATA__", data)


if __name__ == "__main__":
    main()
