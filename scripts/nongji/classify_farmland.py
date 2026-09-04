#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
농지 파이프라인 v1 — 지목 전·답·과 칩을 RemoteCLIP zero-shot 으로
[농지(정상)] vs [형질변경의심] 스크리닝. 비닐하우스는 별도 플래그.

임야 A1 에서 가장 안정적이던 방식(폴리곤 밖 패치토큰 드롭)을 그대로 쓴다.
분류는 farm vs built **2-way softmax** (임야 A1 과 동형).
  farm  = 밭/논/과수원/휴경/수확 후 나지 (겨울 leaf-off 포함)
  built = 건물·주택·창고·태양광·주차·야적·포장·조성 나지·성토

비닐하우스는 별도 그룹이지만 softmax 에 넣지 않는다(겨울 나지가 greenhouse 로 쏠려
확률 질량을 훔치는 붕괴가 있었음). raw 코사인 gh_sim 만 참고 컬럼으로 남기고,
gh_sim 이 built·farm 을 모두 확실히 웃돌 때(+ 여유 margin)만 '비닐하우스검토' 로 표시.

라벨:  p_built >= built-tau                          -> '형질변경의심'
       elif gh_sim >= gh-min AND gh_sim - max(built_sim, farm_sim) >= gh-margin
                                                     -> '비닐하우스검토'
       else                                         -> '농지'
dominant_built = built 하위프롬프트 중 raw 코사인 최댓값 (무엇처럼 보이나)

입력 : data/chips/37714092/*.png (mask, RGBA) + index.csv (jimok 컬럼)

사용:
    python scripts/nongji/classify_farmland.py --cpu
    python scripts/nongji/classify_farmland.py --cpu --jimok 전 --limit 50
    python scripts/nongji/classify_farmland.py --cpu --built-tau 0.4 --no-split
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.remoteclip_backbone import (  # noqa: E402
    ARCH, load_model, group_embeddings, load_index, chip_inputs,
    interior_patches, encode_kept,
)

DEF_CHIPS = Path("data/chips/37714092")
DEF_OUT = Path("data/remoteclip/37714092_nongji_farmland.csv")
DEF_SPLIT = Path("data/output")
JIMOKS = ["전", "답", "과"]

GROUPS = {
    "farm": [
        "a bare plowed farm field",
        "a dry agricultural field with soil",
        "a rice paddy field",
        "farmland with crop rows and furrows",
        "a fallow field with dry vegetation",
        "a harvested field with bare soil in winter",
        "an orchard with rows of bare fruit trees",
        "cultivated farmland",
    ],
    "built": [
        "a building with a roof",
        "houses in a residential area",
        "a large warehouse or factory building",
        "rows of solar photovoltaic panels",
        "a parking lot with parked vehicles",
        "an open yard with stored materials, containers, or equipment",
        "a paved or concrete-covered lot",
        "graded bare ground for construction",
        "a large mound of piled earth or fill soil",
    ],
    "greenhouse": [
        "rows of plastic greenhouses",
        "long white plastic-covered greenhouse tunnels",
        "a cluster of vinyl greenhouses",
    ],
}
LABEL_DIR = {"농지": "농지_정상", "형질변경의심": "농지_형질변경의심", "비닐하우스검토": "농지_비닐하우스"}


def ranked_montage(rows, chips_dir, out_path, cols=8, cell=200, topn=72):
    rr = sorted(rows, key=lambda r: -r["p_built"])[:topn]
    if not rr:
        return
    rn = (len(rr) + cols - 1) // cols
    cap = 22
    cv = Image.new("RGB", (cols * cell, rn * (cell + cap)), (26, 26, 26))
    dr = ImageDraw.Draw(cv)
    color = {"형질변경의심": (255, 90, 90), "비닐하우스검토": (120, 180, 255), "농지": (150, 200, 150)}
    for i, r in enumerate(rr):
        im = Image.open(chips_dir / r["chip"]).convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (26, 26, 26, 255)), im)
        im = im.convert("RGB").resize((cell, cell))
        x, y = (i % cols) * cell, (i // cols) * (cell + cap)
        cv.paste(im, (x, y))
        dr.rectangle([x, y + cell, x + cell, y + cell + cap], fill=(0, 0, 0))
        dr.text((x + 3, y + cell + 5),
                f'{r["jimok"]} b{r["p_built"]:.2f} {r["dominant_built"][:16]}',
                fill=color.get(r["label"], (200, 200, 200)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv.save(out_path)
    print(f"랭킹 몽타주 → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chips", type=Path, default=DEF_CHIPS)
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    ap.add_argument("--jimok", default=",".join(JIMOKS), help="쉼표구분 지목부호 (기본 전,답,과)")
    ap.add_argument("--built-tau", type=float, default=0.60, help="p_built(2-way) 임계 -> 형질변경의심")
    ap.add_argument("--gh-min", type=float, default=0.24, help="gh_sim 최소치")
    ap.add_argument("--gh-margin", type=float, default=0.03, help="gh_sim - max(built_sim,farm_sim) 여유")
    ap.add_argument("--keep-tau", type=float, default=0.5, help="패치 유지 커버리지 임계 (A1)")
    ap.add_argument("--min-keep", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개만")
    ap.add_argument("--sample", type=int, default=0, help="전체에서 무작위 N개 (limit 대신)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--montage-n", type=int, default=72)
    ap.add_argument("--split-dir", type=Path, default=DEF_SPLIT)
    ap.add_argument("--no-split", dest="split", action="store_false")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    meta = load_index(args.index or (args.chips / "index.csv"))
    jimoks = [j.strip() for j in args.jimok.split(",") if j.strip()]

    files = sorted(p for p in args.chips.glob("*.png") if not p.name.startswith("_"))
    files = [f for f in files if meta.get(f.name, {}).get("jimok") in jimoks]
    if args.sample and args.sample < len(files):
        import random
        files = sorted(random.Random(args.seed).sample(files, args.sample))
    elif args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"지목 {jimoks} 칩 없음: {args.chips}")

    print(f"모델 {ARCH} | {device} | 지목 {jimoks} 칩 {len(files)} | "
          f"built_tau={args.built_tau} gh_min={args.gh_min} gh_margin={args.gh_margin} "
          f"keep_tau={args.keep_tau}")
    model, preprocess, tokenizer = load_model(device)
    per_group, sub_names, E_sub = group_embeddings(model, tokenizer, device, GROUPS)
    G2 = torch.stack([per_group["farm"], per_group["built"]])                            # 2-way
    E_farm_m, E_built_m, E_gh_m = per_group["farm"], per_group["built"], per_group["greenhouse"]
    E_built = E_sub[[i for i, (g, _) in enumerate(sub_names) if g == "built"]]
    built_phrases = [p for (g, p) in sub_names if g == "built"]
    ls = model.logit_scale.exp().item()
    v = model.visual

    rows = []
    for i, f in enumerate(files):
        px, alpha = chip_inputs(f, preprocess, device)
        keep, _ = interior_patches(alpha, args.keep_tau, args.min_keep)
        ie = encode_kept(v, px, torch.tensor(keep, dtype=torch.long, device=device))    # [1,D]
        p = (ls * ie @ G2.T).softmax(-1).cpu().numpy()[0]                                # farm/built
        farm_sim = float(ie @ E_farm_m)
        built_sim = float(ie @ E_built_m)
        gh_sim = float(ie @ E_gh_m)
        db = built_phrases[int((ie @ E_built.T).argmax().item())]
        m = meta.get(f.name, {})
        p_farm, p_built = float(p[0]), float(p[1])
        if p_built >= args.built_tau:
            label = "형질변경의심"
        elif gh_sim >= args.gh_min and gh_sim - max(built_sim, farm_sim) >= args.gh_margin:
            label = "비닐하우스검토"
        else:
            label = "농지"
        rows.append({
            "chip": f.name, "pnu": m.get("pnu", ""), "jibun": m.get("jibun", ""),
            "jimok": m.get("jimok", ""), "area_m2": m.get("area_m2", ""),
            "valid_ratio": m.get("valid_ratio", ""), "n_patch_kept": int(len(keep)),
            "p_farm": round(p_farm, 4), "p_built": round(p_built, 4),
            "farm_sim": round(farm_sim, 4), "built_sim": round(built_sim, 4),
            "gh_sim": round(gh_sim, 4), "label": label, "dominant_built": db,
        })
        print(f"  ...{i+1}/{len(files)}", end="\r")
    print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"CSV → {args.out}\n")

    n = len(rows)
    print("── 결과 (전체) ──")
    for k, c in Counter(r["label"] for r in rows).most_common():
        print(f"  {k:<12} {c:>4}  ({100*c/n:.1f}%)")
    print("\n── 지목별 ──")
    for jm in jimoks:
        sub = [r for r in rows if r["jimok"] == jm]
        if not sub:
            continue
        cc = Counter(r["label"] for r in sub)
        print(f"  {jm} ({len(sub)}): " + " · ".join(f"{k} {v}" for k, v in cc.most_common()))

    print("\n── built_tau 스윕 ──")
    for bt in (0.35, 0.40, 0.45, 0.50, 0.60):
        print(f"  {bt:.2f} → 형질변경의심 {sum(1 for r in rows if r['p_built'] >= bt)}")

    susp = [r for r in rows if r["label"] == "형질변경의심"]
    if susp:
        print("\n── 형질변경의심 dominant_built 분포 ──")
        for k, c in Counter(r["dominant_built"] for r in susp).most_common():
            print(f"  {c:>4}  {k}")
        print("\n── p_built 최고 20 ──")
        for r in sorted(rows, key=lambda r: -r["p_built"])[:20]:
            print(f"  b={r['p_built']:.3f} gh={r['gh_sim']:.2f}  {r['jimok']} {r['jibun']:<12} "
                  f"{float(r['area_m2']):>9.0f}㎡ n={r['n_patch_kept']:>2} → {r['label']}  {r['dominant_built']}")

    ranked_montage(rows, args.chips, args.out.with_name(args.out.stem + "_ranked.png"),
                   topn=args.montage_n)

    if args.split:
        for sub in LABEL_DIR.values():
            d = args.split_dir / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        for r in rows:
            shutil.copy2(args.chips / r["chip"], args.split_dir / LABEL_DIR[r["label"]] / r["chip"])
        shutil.copy2(args.out, args.split_dir / args.out.name)
        print("\n분리 저장:")
        for lab, sub in LABEL_DIR.items():
            print(f"  {args.split_dir / sub}  ({sum(1 for r in rows if r['label'] == lab)})")


if __name__ == "__main__":
    main()
