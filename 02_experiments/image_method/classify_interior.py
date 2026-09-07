#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
필지 내부만 인코딩 (interior-only) — RemoteCLIP ViT 에서 필지 폴리곤 '바깥' 패치토큰을 드롭.
실험 기록에서는 'A1' 로 지칭.

가설: mask 칩의 검정 패딩 토큰이 CLS 임베딩을 오염시킨다.
      -> conv/patchify 후, 폴리곤 커버리지 >= tau 인 패치토큰 + CLS 만 남겨
         transformer 를 태우면(=전 블록에서 외부 key 마스킹과 등가) 임베딩이
         '필지 내부 장면'만 반영한다.

입력  : 01_preprocess/chips/37714092_mask/*.png  (RGBA. alpha=폴리곤 마스크)
스코어: classify_wholeimage 의 프롬프트/템플릿/2-way softmax 그대로.
        이미지 임베딩 계산 경로만 교체 -> whole-image(전체 49토큰) 과 직접 비교.
docs/experiments.md §3·§6c 참조. 운영 대응 = 03_pipeline encoder="interior".

사용:
    python 02_experiments/image_method/classify_interior.py --cpu
    python 02_experiments/image_method/classify_interior.py --cpu --tau 0.5 --min-keep 4
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.remoteclip_backbone import (  # noqa: E402
    ARCH_GRID, load_model, subprompt_embeddings, load_index, patch_coverage, encode_kept,
    chip_inputs,
)
from common.prompts import PROMPT_SETS, load_prompt_pair as load_prompts_file  # noqa: E402
from common.paths import chip_dir, EXPERIMENT_RESULTS  # noqa: E402

DEF_CHIPS = chip_dir("mask")
DEF_OUT = EXPERIMENT_RESULTS / "image_method" / "interior.csv"
V1_CSV = EXPERIMENT_RESULTS / "image_method" / "wholeimage.csv"
JIMOK = "임"


def load_v1(p: Path) -> dict[str, dict]:
    if not p.exists():
        return {}
    with open(p, encoding="utf-8-sig") as f:
        return {r["chip"]: r for r in csv.DictReader(f)}


def ranked_montage(rows, chips_dir, out_path, cols=8, cell=200, topn=60):
    rr = sorted(rows, key=lambda r: r["p_forest"])[:topn]
    if not rr:
        return
    rn = (len(rr) + cols - 1) // cols
    cap = 20
    cv = Image.new("RGB", (cols * cell, rn * (cell + cap)), (26, 26, 26))
    dr = ImageDraw.Draw(cv)
    for i, r in enumerate(rr):
        im = Image.open(chips_dir / r["chip"]).convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (26, 26, 26, 255)), im)
        im = im.convert("RGB").resize((cell, cell))
        x, y = (i % cols) * cell, (i // cols) * (cell + cap)
        cv.paste(im, (x, y))
        dr.rectangle([x, y + cell, x + cell, y + cell + cap], fill=(0, 0, 0))
        chg = "" if r["v1_label"] == r["label"] else f'  (v1:{r["v1_label"]})'
        dr.text((x + 3, y + cell + 4),
                f'{r["p_forest"]:.2f} k{r["n_patch_kept"]} {r["closest_neg"][:15]}{chg}',
                fill=(255, 90, 90) if r["label"] == "형질변경의심" else (150, 200, 150))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv.save(out_path)
    print(f"랭킹 몽타주 → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chips", type=Path, default=DEF_CHIPS)
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    ap.add_argument("--tau", type=float, default=0.5, help="패치 유지 커버리지 임계 (>= tau 유지)")
    ap.add_argument("--any-overlap", action="store_true",
                    help="커버리지가 조금이라도(>0) 겹치는 패치는 모두 유지 (--tau 무시)")
    ap.add_argument("--min-keep", type=int, default=4, help="최소 유지 패치 수(상위 커버리지)")
    ap.add_argument("--forest-tau", type=float, default=0.5, help="p_forest 임계")
    ap.add_argument("--real", action="store_true",
                    help="real 칩(폴리곤 밖도 실영상) - alpha 무시하고 원본 RGB 사용")
    ap.add_argument("--prompts", choices=list(PROMPT_SETS), default="v1",
                    help="v1=기존 7/14  v2=descriptor 기반 6/7 (형질변경 특화)")
    ap.add_argument("--prompts-file", type=Path, default=None,
                    help='JSON {"positive":[...],"negative":[...]}. 주면 --prompts 무시')
    ap.add_argument("--arch", choices=list(ARCH_GRID), default="ViT-B-32",
                    help="ViT-B-32(7x7=49패치) vs ViT-L-14(16x16=256패치, 걸침 얇음, 느림)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--montage-n", type=int, default=60)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    meta = load_index(args.chips / "index.csv")
    v1 = load_v1(V1_CSV)

    files = sorted(p for p in args.chips.glob("*.png") if not p.name.startswith("_"))
    files = [f for f in files if meta.get(f.name, {}).get("jimok") == JIMOK]
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"임야 칩 없음: {args.chips}")

    if args.prompts_file:
        POSITIVE, NEGATIVE = load_prompts_file(args.prompts_file)
        pset = f"file:{args.prompts_file.stem}"
    else:
        POSITIVE, NEGATIVE = PROMPT_SETS[args.prompts]
        pset = args.prompts
    grid = ARCH_GRID[args.arch]
    print(f"A1 | {args.arch} ({grid}x{grid}) | {device} | 임야 칩 {len(files)} | "
          f"prompts={pset} ({len(POSITIVE)}/{len(NEGATIVE)}) | tau={args.tau} "
          f"min_keep={args.min_keep} forest_tau={args.forest_tau} real={args.real} "
          f"any_overlap={args.any_overlap}")
    model, preprocess, tokenizer = load_model(device, args.arch)
    v = model.visual
    E_pos = subprompt_embeddings(model, tokenizer, device, POSITIVE).mean(0)
    E_neg = subprompt_embeddings(model, tokenizer, device, NEGATIVE).mean(0)
    E_grp = torch.stack([E_pos, E_neg])
    E_grp = E_grp / E_grp.norm(dim=-1, keepdim=True)
    E_all = subprompt_embeddings(model, tokenizer, device,
                                 list(POSITIVE) + list(NEGATIVE))
    logit_scale = model.logit_scale.exp().item()
    n_pos = len(POSITIVE)

    rows = []
    kepts = []
    for i, f in enumerate(files):
        px, alpha = chip_inputs(f, preprocess, device, composite_black=not args.real)

        cov = patch_coverage(alpha, grid)
        keep = np.where(cov > 0)[0] if args.any_overlap else np.where(cov >= args.tau)[0]
        if len(keep) < args.min_keep:
            keep = np.argsort(-cov)[:args.min_keep]
        keep = np.sort(keep)
        kepts.append(len(keep))
        keep_idx = torch.tensor(keep, dtype=torch.long, device=device)

        ie = encode_kept(v, px, keep_idx)                                    # [1,512]
        p = (logit_scale * ie @ E_grp.T).softmax(-1).cpu().numpy()[0]        # (2,)
        s = (ie @ E_all.T).cpu().numpy()[0]                                  # (n_pos+n_neg,)
        m = meta.get(f.name, {})
        p_forest = float(p[0])
        label = "임야" if p_forest >= args.forest_tau else "형질변경의심"
        rows.append({
            "chip": f.name, "pnu": m.get("pnu", ""), "jibun": m.get("jibun", ""),
            "area_m2": m.get("area_m2", ""), "valid_ratio": m.get("valid_ratio", ""),
            "n_patch_kept": int(len(keep)),
            "p_forest": round(p_forest, 4), "p_notforest": round(float(p[1]), 4),
            "label": label,
            "closest_pos": POSITIVE[int(s[:n_pos].argmax())],
            "closest_neg": NEGATIVE[int(s[n_pos:].argmax())],
            "v1_p_forest": v1.get(f.name, {}).get("p_forest", ""),
            "v1_label": v1.get(f.name, {}).get("label", ""),
        })
        print(f"  ...{i+1}/{len(files)}", end="\r")
    print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"CSV → {args.out}\n")

    n = len(rows)
    kk = np.array(kepts)
    print(f"유지 패치 수: min {kk.min()} · median {int(np.median(kk))} · "
          f"mean {kk.mean():.1f} · max {kk.max()}  (전체 49)")
    print(f"  min_keep 폴백 발동: {(kk <= args.min_keep).sum()} 필지\n")

    lab = Counter(r["label"] for r in rows)
    print("── A1 결과 ──")
    for k, val in lab.most_common():
        print(f"  {k:<10} {val:>4}  ({100*val/n:.1f}%)")
    print("  참고 v1(mask 전체토큰): 임야 365 / 형질변경의심 63")
    print("       v2(context 전체) : 임야 246 / 형질변경의심 182\n")

    if v1:
        both = [r for r in rows if r["v1_label"]]
        same = sum(r["label"] == r["v1_label"] for r in both)
        flip_to_susp = [r for r in both if r["v1_label"] == "임야" and r["label"] == "형질변경의심"]
        flip_to_forest = [r for r in both if r["v1_label"] == "형질변경의심" and r["label"] == "임야"]
        print(f"── v1 대비 ({len(both)}필지 매칭) ──")
        print(f"  라벨 동일           : {same}  ({100*same/len(both):.1f}%)")
        print(f"  임야 → 형질변경의심 : {len(flip_to_susp)}")
        print(f"  형질변경의심 → 임야 : {len(flip_to_forest)}")
        dp = np.array([float(r["p_forest"]) - float(r["v1_p_forest"])
                       for r in both if r["v1_p_forest"]])
        print(f"  Δp_forest (A1 − v1) : mean {dp.mean():+.3f}  median {np.median(dp):+.3f}\n")
        for tag, fl in [("임야→형질변경의심", flip_to_susp), ("형질변경의심→임야", flip_to_forest)]:
            if fl:
                print(f"  [{tag}] 상위 10")
                for r in sorted(fl, key=lambda r: abs(float(r["p_forest"]) - float(r["v1_p_forest"] or 0)),
                                reverse=True)[:10]:
                    print(f"    {r['jibun']:<12} {float(r['area_m2']):>9.0f}㎡ vr={r['valid_ratio']} "
                          f"k={r['n_patch_kept']:>2}  v1 p={r['v1_p_forest']} → A1 p={r['p_forest']}  "
                          f"({r['closest_neg']})")
                print()

    print("── A1 p_forest 최저 15 ──")
    for r in sorted(rows, key=lambda r: r["p_forest"])[:15]:
        mk = "" if r["v1_label"] == r["label"] else f"  ←v1:{r['v1_label']}"
        print(f"  {r['p_forest']:.3f}  {r['jibun']:<12} {float(r['area_m2']):>9.0f}㎡  "
              f"k={r['n_patch_kept']:>2}  → {r['closest_neg']}{mk}")

    ranked_montage(rows, args.chips, args.out.with_name(args.out.stem + "_ranked.png"),
                   topn=args.montage_n)


if __name__ == "__main__":
    main()
