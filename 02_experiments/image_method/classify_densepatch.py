#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
패치별 dense 점수 (dense per-patch) — MaskCLIP식 패치 단위 점수 + 내부 max/fraction 집계.
실험 기록에서는 'C2' 로 지칭. (결론: 실패 — 패치 코사인이 좁게 뭉쳐 built_frac 최대 0.22)

'필지 내부만 인코딩'(interior/A1)의 한계: 큰 필지가 절반만 전환돼도 CLS 평균이 "숲"으로 수렴.
여기서는 CLS를 안 쓰고 패치 하나하나를 텍스트공간으로 뽑아(patch_vec) 각각 built/forest
점수를 매긴 뒤, 폴리곤 내부에서 **fraction / max** 로 집계 → "필지의 몇 %가
건물처럼 보이나". 부분 전환이 평균에 안 묻힌다.

파이프라인:
  1) _embeds 후 폴리곤 커버리지>=keep-tau 패치 + CLS 만 남김 (interior 방식)
  2) resblock 0..10 정상 통과
  3) resblock 11: MaskCLIP surgery — q·k attention 제거, value-projection 만 통과
     (--no-last-residual / --last-mlp 로 ClearCLIP 변형 토글)
  4) ln_post -> proj -> 내부 패치별 512d 벡터, L2 정규화
  5) 패치별 p_built = softmax(sim_pos, sim_neg)[neg]
  6) 커버리지 가중 집계: built_frac, max_built, built_p90, mean_built
  7) built_frac>=frac-tau  또는  max_built>=max-tau  -> 형질변경의심

입력 : 01_preprocess/chips/37714092_mask/*.png (RGBA alpha=폴리곤). 프롬프트는
       classify_wholeimage 와 동일 -> whole-image/interior 와 직접 비교.
docs/experiments.md §4 참조.

사용:
    python 02_experiments/image_method/classify_densepatch.py --cpu
    python 02_experiments/image_method/classify_densepatch.py --cpu --frac-tau 0.30 --max-tau 0.85 --last-mlp
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
    ARCH_GRID, load_model, subprompt_embeddings, load_index, patch_coverage, chip_inputs,
)
from common.prompts import PROMPT_SETS  # noqa: E402
from common.paths import chip_dir, EXPERIMENT_RESULTS  # noqa: E402

DEF_CHIPS = chip_dir("mask")
DEF_OUT = EXPERIMENT_RESULTS / "image_method" / "densepatch.csv"
V1_CSV = EXPERIMENT_RESULTS / "image_method" / "wholeimage.csv"
A1_CSV = EXPERIMENT_RESULTS / "image_method" / "interior.csv"
JIMOK = "임"


def load_csv(p: Path) -> dict[str, dict]:
    if not p.exists():
        return {}
    with open(p, encoding="utf-8-sig") as f:
        return {r["chip"]: r for r in csv.DictReader(f)}


@torch.no_grad()
def dense_patch_vecs(v, px, keep_idx, keep_residual=True, last_mlp=False):
    """px[1,3,224,224], keep_idx=유지 패치(0..48). 반환: 내부 패치별 [K,512] (L2 정규화)."""
    x = v._embeds(px)                                   # [1,50,768]
    sel = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device), keep_idx + 1])
    x = x.index_select(1, sel)                          # [1,1+K,768]  (CLS + 내부패치)

    blocks = v.transformer.resblocks
    for blk in blocks[:-1]:
        x = blk(x)                                      # 정상 attention

    last = blocks[-1]
    h = last.ln_1(x)                                    # [1,1+K,768]
    W = last.attn.in_proj_weight
    b = last.attn.in_proj_bias
    D = last.attn.embed_dim
    Wv, bv = W[2 * D:3 * D, :], b[2 * D:3 * D]          # value projection
    val = torch.nn.functional.linear(h, Wv, bv)
    attn_out = last.attn.out_proj(val)                  # q·k·softmax 없음
    x = (x + attn_out) if keep_residual else attn_out
    if last_mlp:
        x = x + last.mlp(last.ln_2(x))

    feat = v.ln_post(x)
    pv = feat[:, 1:, :] @ v.proj                        # [1,K,512]  (CLS 버림)
    pv = pv / pv.norm(dim=-1, keepdim=True)
    return pv[0]                                        # [K,512]


def wpercentile(vals, w, q):
    o = np.argsort(vals)
    vals, w = vals[o], w[o]
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), c, vals))


def montage(rows, chips_dir, out_path, cols=8, cell=200, topn=60):
    rr = sorted(rows, key=lambda r: (-r["built_frac"], -r["max_built"]))[:topn]
    rn = (len(rr) + cols - 1) // cols
    cap = 22
    cv = Image.new("RGB", (cols * cell, rn * (cell + cap)), (26, 26, 26))
    dr = ImageDraw.Draw(cv)
    for i, r in enumerate(rr):
        im = Image.open(chips_dir / r["chip"]).convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (26, 26, 26, 255)), im)
        im = im.convert("RGB").resize((cell, cell))
        x, y = (i % cols) * cell, (i // cols) * (cell + cap)
        cv.paste(im, (x, y))
        dr.rectangle([x, y + cell, x + cell, y + cell + cap], fill=(0, 0, 0))
        d = "" if r["a1_label"] == r["label"] else f'  a1:{r["a1_label"]}'
        dr.text((x + 3, y + cell + 5),
                f'bf{r["built_frac"]:.2f} mx{r["max_built"]:.2f} {r["dominant_neg"][:13]}{d}',
                fill=(255, 90, 90) if r["label"] == "형질변경의심" else (150, 200, 150))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv.save(out_path)
    print(f"랭킹 몽타주 → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chips", type=Path, default=DEF_CHIPS)
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    ap.add_argument("--arch", choices=list(ARCH_GRID), default="ViT-B-32")
    ap.add_argument("--prompts", choices=list(PROMPT_SETS), default="v1")
    ap.add_argument("--real", action="store_true",
                    help="real 칩(폴리곤 밖도 실영상) - alpha 무시하고 원본 RGB")
    ap.add_argument("--a1-csv", type=Path, default=None, help="비교용 A1 결과 CSV")
    ap.add_argument("--keep-tau", type=float, default=0.5, help="패치 유지 커버리지 임계 (A1)")
    ap.add_argument("--min-keep", type=int, default=4)
    ap.add_argument("--frac-tau", type=float, default=0.35, help="built_frac 임계")
    ap.add_argument("--max-tau", type=float, default=0.85, help="max_built 임계")
    ap.add_argument("--patch-thr", type=float, default=0.5, help="패치를 built로 셀 p_built 임계")
    ap.add_argument("--no-last-residual", dest="last_residual", action="store_false")
    ap.add_argument("--last-mlp", action="store_true", help="마지막 블록 FFN 유지(기본 skip)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--montage-n", type=int, default=60)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    grid = ARCH_GRID[args.arch]
    POSITIVE, NEGATIVE = PROMPT_SETS[args.prompts]
    meta = load_index(args.chips / "index.csv")
    v1, a1 = load_csv(V1_CSV), load_csv(args.a1_csv or A1_CSV)

    files = sorted(p for p in args.chips.glob("*.png") if not p.name.startswith("_"))
    files = [f for f in files if meta.get(f.name, {}).get("jimok") == JIMOK]
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"임야 칩 없음: {args.chips}")

    print(f"C2 | {args.arch} ({grid}x{grid}) MaskCLIP surgery | {device} | 임야 칩 {len(files)} | "
          f"prompts={args.prompts} real={args.real} keep_tau={args.keep_tau} "
          f"frac_tau={args.frac_tau} max_tau={args.max_tau} "
          f"last_residual={args.last_residual} last_mlp={args.last_mlp}")
    model, preprocess, tokenizer = load_model(device, args.arch)
    v = model.visual
    E_pos = subprompt_embeddings(model, tokenizer, device, POSITIVE).mean(0)
    E_neg = subprompt_embeddings(model, tokenizer, device, NEGATIVE).mean(0)
    E_pos = E_pos / E_pos.norm()
    E_neg = E_neg / E_neg.norm()
    E_negsub = subprompt_embeddings(model, tokenizer, device, list(NEGATIVE))
    ls = model.logit_scale.exp().item()

    rows, npatch = [], []
    for i, f in enumerate(files):
        px, alpha = chip_inputs(f, preprocess, device, composite_black=not args.real)

        cov = patch_coverage(alpha, grid)
        keep = np.where(cov >= args.keep_tau)[0]
        if len(keep) < args.min_keep:
            keep = np.argsort(-cov)[:args.min_keep]
        keep = np.sort(keep)
        w = cov[keep].astype(np.float64)
        if w.sum() < 0.5:
            w = np.ones_like(w)
        npatch.append(len(keep))

        pv = dense_patch_vecs(v, px, torch.tensor(keep, dtype=torch.long, device=device),
                              args.last_residual, args.last_mlp)          # [K,512]
        sp = (pv @ E_pos).cpu().numpy()
        sn = (pv @ E_neg).cpu().numpy()
        logits = np.stack([ls * sp, ls * sn], 1)
        logits -= logits.max(1, keepdims=True)
        e = np.exp(logits)
        p_built = e[:, 1] / e.sum(1)                                      # [K]

        W = w.sum()
        built_frac = float((w * (p_built > args.patch_thr)).sum() / W)
        mean_built = float((w * p_built).sum() / W)
        max_built = float(p_built.max())
        p90 = wpercentile(p_built, w, 0.9)
        # 가장 built 인 패치의 세부 클래스
        di = int(p_built.argmax())
        dom = NEGATIVE[int((pv[di] @ E_negsub.T).argmax().item())]

        label = ("형질변경의심" if (built_frac >= args.frac_tau or max_built >= args.max_tau)
                 else "임야")
        m = meta.get(f.name, {})
        rows.append({
            "chip": f.name, "pnu": m.get("pnu", ""), "jibun": m.get("jibun", ""),
            "area_m2": m.get("area_m2", ""), "valid_ratio": m.get("valid_ratio", ""),
            "n_patch": len(keep),
            "built_frac": round(built_frac, 3), "mean_built": round(mean_built, 3),
            "max_built": round(max_built, 3), "built_p90": round(p90, 3),
            "label": label, "dominant_neg": dom,
            "v1_label": v1.get(f.name, {}).get("label", ""),
            "a1_label": a1.get(f.name, {}).get("label", ""),
            "a1_p_forest": a1.get(f.name, {}).get("p_forest", ""),
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
    npa = np.array(npatch)
    print(f"내부 패치 수: min {npa.min()} · median {int(np.median(npa))} · max {npa.max()}  (49중)\n")

    lab = Counter(r["label"] for r in rows)
    print("── C2 결과 ──")
    for k, val in lab.most_common():
        print(f"  {k:<10} {val:>4}  ({100*val/n:.1f}%)")
    print("  참고 v1: 임야365/의심63 · A1: 임야382/의심46 · v2(context): 임야246/의심182\n")

    # frac_tau 스윕 (max_tau 고정)
    print("── built_frac 임계 스윕 (max_tau %.2f 유지) ──" % args.max_tau)
    for ft in (0.20, 0.30, 0.35, 0.40, 0.50):
        c = sum(1 for r in rows if r["built_frac"] >= ft or r["max_built"] >= args.max_tau)
        print(f"  frac_tau {ft:.2f} → 형질변경의심 {c}")
    print()

    if a1:
        both = [r for r in rows if r["a1_label"]]
        same = sum(r["label"] == r["a1_label"] for r in both)
        rec = [r for r in both if r["a1_label"] == "임야" and r["label"] == "형질변경의심"]
        lost = [r for r in both if r["a1_label"] == "형질변경의심" and r["label"] == "임야"]
        print(f"── A1 대비 ({len(both)}) ──")
        print(f"  동일 {same} ({100*same/len(both):.0f}%) · A1임야→C2의심 {len(rec)} · A1의심→C2임야 {len(lost)}\n")
        print("  [A1은 임야, C2가 형질변경의심으로 회수] built_frac 순 상위 15")
        for r in sorted(rec, key=lambda r: -r["built_frac"])[:15]:
            print(f"    {r['jibun']:<12} {float(r['area_m2']):>9.0f}㎡ vr={r['valid_ratio']} "
                  f"n={r['n_patch']:>2}  bf={r['built_frac']:.2f} mx={r['max_built']:.2f} "
                  f"(A1 p_forest={r['a1_p_forest']}) → {r['dominant_neg']}")
        print()

    watch = ["산18-1임", "산43-1임", "산85임", "산87-2임", "502-24 임", "산10-27 임",
             "산83임", "산89임"]
    print("── 관심 필지 (A1이 놓쳤던 큰 부분전환 + 확실한 케이스) ──")
    for jb in watch:
        r = next((x for x in rows if x["jibun"] == jb), None)
        if r:
            print(f"  {jb:<12} bf={r['built_frac']:.2f} mx={r['max_built']:.2f} "
                  f"p90={r['built_p90']:.2f} n={r['n_patch']:>2} → {r['label']}  "
                  f"(v1={r['v1_label']} a1={r['a1_label']}) {r['dominant_neg']}")
    print()

    print("── built_frac 최고 15 ──")
    for r in sorted(rows, key=lambda r: (-r["built_frac"], -r["max_built"]))[:15]:
        d = "" if r["a1_label"] == r["label"] else f"  ←a1:{r['a1_label']}"
        print(f"  bf={r['built_frac']:.2f} mx={r['max_built']:.2f}  {r['jibun']:<12} "
              f"{float(r['area_m2']):>9.0f}㎡ n={r['n_patch']:>2} → {r['dominant_neg']}{d}")

    montage(rows, args.chips, args.out.with_name(args.out.stem + "_ranked.png"),
            topn=args.montage_n)


if __name__ == "__main__":
    main()
