#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
임야 파이프라인 v1 — 지목 '임' 칩을 RemoteCLIP zero-shot 으로
[임야] vs [형질변경 의심] 2진 분류. whole-image CLS 임베딩.

positive(임야) = 수림만. 낙엽기 활엽수림 포함. 자연 암석지·황무지·산지 초지 제외.
negative       = 벌채/개간 나지, 건물, 도로, 주차/야적, 태양광, 경작지,
                 비닐하우스, 묘지(봉분), 초지, 암석 나지.

점수: 하위프롬프트 = (문구 x 템플릿) 평균. 그룹 임베딩 = 하위프롬프트 평균.
      이미지 vs [E_pos, E_neg] 2-way softmax. p_forest >= tau -> '임야'.

사용:
    python scripts/imya/classify_binary.py --cpu
    python scripts/imya/classify_binary.py --cpu --tau 0.5 --limit 50
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.remoteclip_backbone import ARCH, load_model, subprompt_embeddings, load_index  # noqa: E402

DEF_CHIPS = Path("data/chips/37714092")
DEF_OUT = Path("data/remoteclip/37714092_imya_binary.csv")
JIMOK = "임"

POSITIVE = [
    "a forest",
    "dense green woodland",
    "a tree-covered hill",
    "a forested mountain slope",
    "a wooded area with many trees",
    "leafless deciduous forest in winter",
    "bare brown trees covering a hillside",
]

NEGATIVE = [
    "buildings and rooftops",
    "houses in a residential area",
    "a large warehouse or factory building",
    "a paved road",
    "a parking lot or vehicle storage yard",
    "an open yard with stored materials or containers",
    "bare earth cleared of trees",
    "graded bare construction ground",
    "rows of solar photovoltaic panels",
    "a cultivated crop field",
    "plastic greenhouses",
    "rows of grave mounds on a hillside",
    "a grassy field",
    "bare rocky ground",
]

# --- v2: descriptor 기반 (CuPL/Menon&Vondrick 식). 임야 형질변경에 특화 -------------
POSITIVE_V2 = [
    "a hillside densely covered with tree canopy",
    "a mix of evergreen and leafless deciduous trees on rough terrain",
    "an uncultivated wooded slope with no roads or buildings",
    "forest with irregular natural tree spacing, not planted in rows",
    "a bare brown deciduous forest canopy in winter",
    "a steep forested mountainside",
]

NEGATIVE_V2 = [
    "a patch of bare soil with tree stumps where forest was cleared",
    "graded flat bare earth with tire tracks cut into a wooded hill",
    "a metal-roofed shed or building surrounded by trees",
    "rows of dark solar panels on a cleared hillside",
    "a new dirt or paved road cut through forest",
    "grave mounds on a cleared hillside",
    "a cultivated field or greenhouse on former forest land",
]

PROMPT_SETS = {"v1": (POSITIVE, NEGATIVE), "v2": (POSITIVE_V2, NEGATIVE_V2)}


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


@torch.no_grad()
def classify(model, preprocess, device, files, meta, E_pos, E_neg, E_all,
             pos_names, neg_names, batch, tau):
    logit_scale = model.logit_scale.exp().item()
    E_grp = torch.stack([E_pos.mean(0), E_neg.mean(0)])
    E_grp = E_grp / E_grp.norm(dim=-1, keepdim=True)
    n_pos = len(pos_names)
    rows, done = [], 0
    for grp in chunks(files, batch):
        px = torch.stack([preprocess(Image.open(f).convert("RGB")) for f in grp]).to(device)
        ie = model.encode_image(px).float()
        ie = ie / ie.norm(dim=-1, keepdim=True)
        P = (logit_scale * ie @ E_grp.T).softmax(-1).cpu().numpy()
        S = (ie @ E_all.T).cpu().numpy()
        for f, p, s in zip(grp, P, S):
            m = meta.get(f.name, {})
            p_forest = float(p[0])
            rows.append({
                "chip": f.name, "pnu": m.get("pnu", ""), "jibun": m.get("jibun", ""),
                "jimok": m.get("jimok", ""), "area_m2": m.get("area_m2", ""),
                "valid_ratio": m.get("valid_ratio", ""),
                "p_forest": round(p_forest, 4), "p_notforest": round(float(p[1]), 4),
                "label": "임야" if p_forest >= tau else "형질변경의심",
                "closest_pos": pos_names[int(s[:n_pos].argmax())],
                "closest_pos_cos": round(float(s[:n_pos].max()), 4),
                "closest_neg": neg_names[int(s[n_pos:].argmax())],
                "closest_neg_cos": round(float(s[n_pos:].max()), 4),
            })
        done += len(grp)
        print(f"  ...{done}/{len(files)}", end="\r")
    print()
    return rows


def ranked_montage(rows, chips_dir, out_path, cols=8, cell=200, topn=60):
    rr = sorted(rows, key=lambda r: r["p_forest"])[:topn]
    if not rr:
        return
    rn = (len(rr) + cols - 1) // cols
    cap = 18
    canvas = Image.new("RGB", (cols * cell, rn * (cell + cap)), (26, 26, 26))
    dr = ImageDraw.Draw(canvas)
    for i, r in enumerate(rr):
        im = Image.open(chips_dir / r["chip"]).convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (26, 26, 26, 255)), im)
        im = im.convert("RGB").resize((cell, cell))
        x, y = (i % cols) * cell, (i // cols) * (cell + cap)
        canvas.paste(im, (x, y))
        dr.rectangle([x, y + cell, x + cell, y + cell + cap], fill=(0, 0, 0))
        dr.text((x + 3, y + cell + 3), f'{r["p_forest"]:.2f} {r["closest_neg"][:22]}',
                fill=(255, 90, 90) if r["label"] == "형질변경의심" else (150, 200, 150))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"랭킹 몽타주 → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chips", type=Path, default=DEF_CHIPS)
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    ap.add_argument("--jimok", default=JIMOK)
    ap.add_argument("--prompts", choices=list(PROMPT_SETS), default="v1")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--montage-n", type=int, default=60)
    ap.add_argument("--no-montage", dest="montage", action="store_false")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    meta = load_index(args.index or (args.chips / "index.csv"))

    files = sorted(p for p in args.chips.glob("*.png") if not p.name.startswith("_"))
    files = [f for f in files if meta.get(f.name, {}).get("jimok") == args.jimok]
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"지목 '{args.jimok}' 칩 없음: {args.chips}")

    POSITIVE, NEGATIVE = PROMPT_SETS[args.prompts]
    print(f"모델 {ARCH} | {device} | 지목 '{args.jimok}' 칩 {len(files)} | prompts={args.prompts} "
          f"({len(POSITIVE)}/{len(NEGATIVE)}) | tau={args.tau}")
    model, preprocess, tokenizer = load_model(device)
    E_pos = subprompt_embeddings(model, tokenizer, device, POSITIVE)
    E_neg = subprompt_embeddings(model, tokenizer, device, NEGATIVE)
    E_all = torch.cat([E_pos, E_neg], 0)

    rows = classify(model, preprocess, device, files, meta, E_pos, E_neg, E_all,
                    POSITIVE, NEGATIVE, args.batch, args.tau)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV → {args.out}")

    n = len(rows)
    print("\n── 2진 분류 결과 ──")
    for k, v in Counter(r["label"] for r in rows).most_common():
        print(f"  {k:<10} {v:>4}  ({100*v/n:.1f}%)")

    susp = [r for r in rows if r["label"] == "형질변경의심"]
    if susp:
        print("\n── 형질변경의심 closest_neg 분포 ──")
        for k, v in Counter(r["closest_neg"] for r in susp).most_common():
            print(f"  {v:>4}  {k}")
        print("\n── p_forest 최저 15 ──")
        for r in sorted(rows, key=lambda r: r["p_forest"])[:15]:
            print(f"  {r['p_forest']:.3f}  {r['jibun']:<12} {r['area_m2']:>10}㎡  "
                  f"vr={r['valid_ratio']}  → {r['closest_neg']}")

    if args.montage:
        ranked_montage(rows, args.chips, args.out.with_name(args.out.stem + "_ranked.png"),
                       topn=args.montage_n)


if __name__ == "__main__":
    main()
