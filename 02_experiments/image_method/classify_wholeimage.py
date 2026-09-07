#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전체 이미지 인코딩 (whole-image) — 지목 '임' 칩 전체를 RemoteCLIP 로 한 번에 인코딩(CLS 토큰)해
[임야] vs [형질변경 의심] 2진 분류. 실험 기록에서는 'v1' 로 지칭.

폴리곤 밖(검정 패딩 또는 실영상)까지 포함해 이미지 한 장을 통째로 임베딩한다.
docs/experiments.md §1·§5·§6d 참조. 운영 대응 = 03_pipeline encoder="whole".

positive(임야) = 수림만. 낙엽기 활엽수림 포함. 자연 암석지·황무지·산지 초지 제외.
negative       = 벌채/개간 나지, 건물, 도로, 주차/야적, 태양광, 경작지,
                 비닐하우스, 묘지(봉분), 초지, 암석 나지.

점수: 하위프롬프트 = (문구 x 템플릿) 평균. 그룹 임베딩 = 하위프롬프트 평균.
      이미지 vs [E_pos, E_neg] 2-way softmax. p_forest >= tau -> '임야'.

사용:
    python 02_experiments/image_method/classify_wholeimage.py --cpu
    python 02_experiments/image_method/classify_wholeimage.py --cpu --fill mean --tau 0.93
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.remoteclip_backbone import ARCH, load_model, subprompt_embeddings, load_index  # noqa: E402
from common.prompts import PROMPT_SETS, load_prompt_pair as load_prompts_file  # noqa: E402
from common.encoders import load_rgb as _load_rgb  # noqa: E402
from common.paths import chip_dir, EXPERIMENT_RESULTS  # noqa: E402

DEF_CHIPS = chip_dir("mask")
DEF_OUT = EXPERIMENT_RESULTS / "image_method" / "wholeimage.csv"
JIMOK = "임"


def load_rgb(f, fill_mode="black"):
    return _load_rgb(f, fill_mode)


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


@torch.no_grad()
def classify(model, preprocess, device, files, meta, E_pos, E_neg, E_all,
             pos_names, neg_names, batch, tau, fill_mode="black"):
    logit_scale = model.logit_scale.exp().item()
    E_grp = torch.stack([E_pos.mean(0), E_neg.mean(0)])
    E_grp = E_grp / E_grp.norm(dim=-1, keepdim=True)
    n_pos = len(pos_names)
    rows, done = [], 0
    for grp in chunks(files, batch):
        px = torch.stack([preprocess(load_rgb(f, fill_mode)) for f in grp]).to(device)
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
    ap.add_argument("--prompts-file", type=Path, default=None,
                    help='JSON {"positive":[...],"negative":[...]}. 주면 --prompts 무시')
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--fill", choices=["black", "blackout", "mean", "gray", "inpaint"], default="black",
                    help="폴리곤 밖 채움. black=RGB 그대로 / blackout=alpha로 검정 강제합성 / "
                         "mean=내부평균색 / gray / inpaint")
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

    if args.prompts_file:
        POSITIVE, NEGATIVE = load_prompts_file(args.prompts_file)
        pset = f"file:{args.prompts_file.stem}"
    else:
        POSITIVE, NEGATIVE = PROMPT_SETS[args.prompts]
        pset = args.prompts
    print(f"모델 {ARCH} | {device} | 지목 '{args.jimok}' 칩 {len(files)} | prompts={pset} "
          f"({len(POSITIVE)}/{len(NEGATIVE)}) | tau={args.tau}")
    model, preprocess, tokenizer = load_model(device)
    E_pos = subprompt_embeddings(model, tokenizer, device, POSITIVE)
    E_neg = subprompt_embeddings(model, tokenizer, device, NEGATIVE)
    E_all = torch.cat([E_pos, E_neg], 0)

    rows = classify(model, preprocess, device, files, meta, E_pos, E_neg, E_all,
                    POSITIVE, NEGATIVE, args.batch, args.tau, args.fill)

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
