#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RemoteCLIP zero-shot 으로 필지 칩을 '시각 분류군(visual_class)'으로 분류한다.

- 프롬프트/매핑: data/reference/jimok_taxonomy.json  (28지목 -> visual_class + 영어 프롬프트)
- 가중치: HuggingFace `chendelong/RemoteCLIP` 자동 다운로드 (models/remoteclip/)
- 출력:
    <out>.csv               : 칩별 visual_class 확률 전체 + 예측/일치여부/anomaly
    <out>_visualclass.csv   : 예측 visual_class 분포 (n, %)
    <out>_by_jimok.csv      : 실제 지목 x 예측 교차 (n, 기대 visual_class, 일치율, 최다혼동)
    <out>_ranked.png        : 지목='임' 기준 anomaly 내림차순 몽타주

지목별 %:
  - 실제 지목 분포는 지적도(index.csv 의 jimok)에서 그대로 집계 -> 정확
  - '영상으로 예측한' 분포는 visual_class 수준. 전↔답, 대↔공장↔창고, 도로↔주차장,
    하천↔구거↔유지 등은 영상으로 안 갈라지므로 taxonomy 의 separability 를 함께 본다.

사용:
    python scripts/run_remoteclip.py                          # ViT-B-32, data/chips/37714092
    python scripts/run_remoteclip.py --model ViT-L-14 --half
    python scripts/run_remoteclip.py --anomaly-jimok 임 --threshold 0.5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import open_clip
from huggingface_hub import hf_hub_download

DEF_CHIPS = Path("data/chips/37714092")
DEF_OUT = Path("data/remoteclip/37714092_scores.csv")
DEF_TAX = Path("data/reference/jimok_taxonomy.json")
CKPT_DIR = Path("models/remoteclip")
HF_REPO = "chendelong/RemoteCLIP"
CKPT_FILE = {"RN50": "RemoteCLIP-RN50.pt",
             "ViT-B-32": "RemoteCLIP-ViT-B-32.pt",
             "ViT-L-14": "RemoteCLIP-ViT-L-14.pt"}

# visual_class -> 상위 그룹 (임야 anomaly 판정 및 리포트용)
GROUP = {
    "forest": "natural", "barren_land": "natural", "pasture_grass": "natural",
    "dry_cropland": "agriculture", "paddy_field": "agriculture", "orchard": "agriculture",
    "greenhouse": "developed", "built_residential": "developed", "built_large": "developed",
    "built_small": "developed", "institutional": "developed", "religious": "developed",
    "road": "developed", "railway": "developed", "parking_yard": "developed",
    "embankment": "developed", "bare_construction": "developed", "solar_farm": "developed",
    "water_body": "water", "narrow_channel": "water", "aquaculture": "water",
    "cemetery": "other", "sports_field": "other", "park_greenspace": "other",
    "salt_pond": "other",
}


# 판별력 높은 핵심 클래스 (--core 옵션). 나머지(embankment/narrow_channel/
# built_small/institutional/religious/railway/salt_pond/park/sports 등)는 zero-shot
# 에서 '블랙홀'로 작용해 제외.
CORE_CLASSES = [
    "forest", "barren_land", "pasture_grass",
    "dry_cropland", "paddy_field", "orchard", "greenhouse",
    "built_residential", "built_large", "road", "parking_yard",
    "water_body", "bare_construction", "solar_farm", "cemetery",
]


def load_taxonomy(path: Path, core: bool = False):
    tax = json.loads(path.read_text(encoding="utf-8"))
    templates = tax["_meta"]["templates"]
    vclasses = tax["visual_classes"]                 # name -> [phrases]
    if core:
        vclasses = {k: vclasses[k] for k in CORE_CLASSES if k in vclasses}
    buho2vc = {j["buho"]: j["visual_class"] for j in tax["jimok"]}
    buho2ko = {j["buho"]: j["ko"] for j in tax["jimok"]}
    buho2sep = {j["buho"]: j["separability"] for j in tax["jimok"]}
    return templates, vclasses, buho2vc, buho2ko, buho2sep


def resolve_ckpt(arch: str) -> str:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  가중치: {HF_REPO}/{CKPT_FILE[arch]}")
    return hf_hub_download(HF_REPO, CKPT_FILE[arch], cache_dir=str(CKPT_DIR))


def load_model(arch: str, device: torch.device, half: bool):
    model, _, preprocess = open_clip.create_model_and_transforms(arch)
    tokenizer = open_clip.get_tokenizer(arch)
    try:
        state = torch.load(resolve_ckpt(arch), map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(resolve_ckpt(arch), map_location="cpu")
    msg = model.load_state_dict(state, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        print(f"  ! state_dict missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    model.eval().to(device)
    if half and device.type == "cuda":
        model.half()
    return model, preprocess, tokenizer


@torch.no_grad()
def text_embeddings(model, tokenizer, device, vclasses, templates):
    names = list(vclasses)
    embs = []
    for nm in names:
        prompts = [t.format(p) for p in vclasses[nm] for t in templates]
        e = model.encode_text(tokenizer(prompts).to(device)).float()
        e = e / e.norm(dim=-1, keepdim=True)
        embs.append(e.mean(0))
    E = torch.stack(embs)
    return names, E / E.norm(dim=-1, keepdim=True)


def load_index(p: Path) -> dict[str, dict]:
    if not p.exists():
        return {}
    with open(p, encoding="utf-8-sig") as f:
        return {r["chip"]: r for r in csv.DictReader(f)}


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


@torch.no_grad()
def classify(model, preprocess, names, E, files, meta, device, half, batch,
             buho2vc, anomaly_jimok, threshold):
    logit_scale = model.logit_scale.exp().item()
    rows = []
    done = 0
    for grp in chunks(files, batch):
        px = torch.stack([preprocess(Image.open(f).convert("RGB")) for f in grp]).to(device)
        if half and device.type == "cuda":
            px = px.half()
        ie = model.encode_image(px).float()
        ie = ie / ie.norm(dim=-1, keepdim=True)
        P = (logit_scale * ie @ E.T).softmax(-1).cpu().numpy()
        for f, p in zip(grp, P):
            d = {nm: float(p[i]) for i, nm in enumerate(names)}
            g = defaultdict(float)
            for nm, v in d.items():
                g[GROUP[nm]] += v
            pred = names[int(p.argmax())]
            m = meta.get(f.name, {})
            buho = m.get("jimok", "")
            exp_vc = buho2vc.get(buho, "")
            row = {
                "chip": f.name, "pnu": m.get("pnu", ""), "jibun": m.get("jibun", ""),
                "jimok": buho, "valid_ratio": m.get("valid_ratio", ""),
                "pred_visual_class": pred, "pred_p": round(float(p.max()), 4),
                "expected_visual_class": exp_vc,
                "match": int(pred == exp_vc) if exp_vc else "",
                "p_natural": round(g["natural"], 4), "p_agriculture": round(g["agriculture"], 4),
                "p_developed": round(g["developed"], 4), "p_water": round(g["water"], 4),
                "p_other": round(g["other"], 4),
            }
            if buho == anomaly_jimok:
                row["anomaly_score"] = round(1.0 - g["natural"], 4)
                row["anomaly_flag"] = int((1.0 - g["natural"]) >= threshold)
            else:
                row["anomaly_score"] = ""
                row["anomaly_flag"] = ""
            row.update({f"p_{nm}": round(d[nm], 4) for nm in names})
            rows.append(row)
        done += len(grp)
        print(f"  ...{done}/{len(files)}", end="\r")
    print()
    return rows


def ranked_montage(rows, chips_dir, out_path, cols=8, cell=200, topn=64):
    rr = [r for r in rows if r["anomaly_score"] != ""]
    rr.sort(key=lambda r: r["anomaly_score"], reverse=True)
    rr = rr[:topn]
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
        dr.text((x + 3, y + cell + 3), f'{r["pred_visual_class"]} {r["anomaly_score"]}',
                fill=(255, 90, 90) if r["anomaly_flag"] == 1 else (200, 200, 200))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"랭킹 몽타주 → {out_path}")


def pct_table(counter: Counter, total: int):
    return [(k, v, round(100 * v / total, 2)) for k, v in counter.most_common()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chips", type=Path, default=DEF_CHIPS)
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--taxonomy", type=Path, default=DEF_TAX)
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    ap.add_argument("--model", default="ViT-B-32", choices=list(CKPT_FILE))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--anomaly-jimok", default="임", help="이 지목 필지에 anomaly_score 부여")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--core", action="store_true",
                    help="판별력 높은 15개 클래스만 사용 (zero-shot 안정화)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-montage", dest="montage", action="store_false")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    index_path = args.index or (args.chips / "index.csv")
    templates, vclasses, buho2vc, buho2ko, buho2sep = load_taxonomy(args.taxonomy, args.core)

    files = sorted(p for p in args.chips.glob("*.png") if not p.name.startswith("_"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"칩 PNG 없음: {args.chips}")

    print(f"모델 {args.model} | {device}{' fp16' if args.half and device.type=='cuda' else ''} "
          f"| 칩 {len(files)} | visual_class {len(vclasses)}종")
    model, preprocess, tokenizer = load_model(args.model, device, args.half)
    names, E = text_embeddings(model, tokenizer, device, vclasses, templates)
    meta = load_index(index_path)

    rows = classify(model, preprocess, names, E, files, meta, device, args.half,
                    args.batch, buho2vc, args.anomaly_jimok, args.threshold)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nCSV → {args.out}")

    # ---- 예측 visual_class 분포 ----
    n = len(rows)
    pred_ct = Counter(r["pred_visual_class"] for r in rows)
    vc_path = args.out.with_name(args.out.stem + "_visualclass.csv")
    with open(vc_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["visual_class", "group", "n", "pct"])
        for k, v, p in pct_table(pred_ct, n):
            w.writerow([k, GROUP[k], v, p])
    print(f"\n── 예측 visual_class 분포 (상위 12) ──")
    for k, v, p in pct_table(pred_ct, n)[:12]:
        print(f"  {p:5.1f}%  {v:>5}  {k:<18} [{GROUP[k]}]")

    # ---- 실제 지목 분포 + 지목 x 예측 교차 ----
    have_jimok = [r for r in rows if r["jimok"]]
    if have_jimok:
        act_ct = Counter(r["jimok"] for r in have_jimok)
        by_path = args.out.with_name(args.out.stem + "_by_jimok.csv")
        print(f"\n── 실제 지목 분포 / 예측 일치율 ──")
        print(f"  {'지목':<8}{'n':>6}{'실제%':>8}{'기대 visual_class':>22}{'일치%':>8}{'sep':>7}  최다예측")
        with open(by_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["buho", "ko", "n", "actual_pct", "expected_visual_class",
                        "separability", "match_pct", "top_pred", "top_pred_pct"])
            for buho, cnt in act_ct.most_common():
                sub = [r for r in have_jimok if r["jimok"] == buho]
                exp_vc = buho2vc.get(buho, "")
                mrate = 100 * sum(1 for r in sub if r["match"] == 1) / len(sub)
                tp = Counter(r["pred_visual_class"] for r in sub).most_common(1)[0]
                w.writerow([buho, buho2ko.get(buho, ""), cnt,
                            round(100 * cnt / len(have_jimok), 2), exp_vc,
                            buho2sep.get(buho, ""), round(mrate, 1),
                            tp[0], round(100 * tp[1] / len(sub), 1)])
                print(f"  {buho2ko.get(buho,buho):<8}{cnt:>6}{100*cnt/len(have_jimok):>7.1f}%"
                      f"{exp_vc:>22}{mrate:>7.0f}%{buho2sep.get(buho,''):>7}  "
                      f"{tp[0]} {100*tp[1]/len(sub):.0f}%")
        print(f"\n교차표 CSV → {by_path}")

        # 임야 anomaly 요약
        aj = [r for r in have_jimok if r["jimok"] == args.anomaly_jimok and r["anomaly_flag"] != ""]
        if aj:
            nf = sum(r["anomaly_flag"] for r in aj)
            print(f"\n지목='{args.anomaly_jimok}' anomaly_flag(≥{args.threshold}): {nf}/{len(aj)} "
                  f"({100*nf/len(aj):.1f}%)")

    if args.montage:
        ranked_montage(rows, args.chips, args.out.with_name(args.out.stem + "_ranked.png"))


if __name__ == "__main__":
    main()
