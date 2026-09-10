#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage 2 — 칩 → RemoteCLIP zero-shot 점수.

config.image.encoder ∈ {whole, interior} 로 이미지 임베딩,
config.prompt_file 의 softmax 두 그룹으로 2-way softmax → p_forest / p_screen.
side_flags 그룹(예: greenhouse)은 raw 코사인만 참고 컬럼으로.

산출: <run_dir>/2_scores.csv   (전체 대상 지목 필지)
컬럼: chip,pnu,jibun,jimok,area_m2,valid_ratio,
      p_forest,p_screen, closest_pos,closest_pos_cos, closest_neg,closest_neg_cos,
      [gh_sim, gh_dominant]         # side_flags 있을 때만
  - p_forest = softmax[0] 그룹 확률 (임야/농지 "정상" 쪽)
  - p_screen = positive_group 확률 (형질변경/built 쪽)
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch

from common.remoteclip_backbone import ARCH_GRID, load_model, subprompt_embeddings, load_index
from common.encoders import encode_whole, encode_interior, dense_patch_change
from common.prompts import load_groups


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


@torch.no_grad()
def run(cfg: dict, run_dir: Path, chips_dir: Path, batch: int = 32) -> Path:
    img = cfg["image"]
    encoder = img["encoder"]
    fill = img.get("fill", "black")
    chip_mode = img["chip_mode"]
    arch = img.get("arch", "ViT-B-32")
    jimok = cfg["target"]["jimok"]
    keep_tau = img.get("keep_tau", 0.5)
    min_keep = img.get("min_keep", 4)
    patch_thr = img.get("patch_thr", 0.5)
    last_mlp = bool(img.get("last_mlp", False))

    spec = load_groups(Path(cfg["prompt_file"]))
    groups, softmax_g, pos_g = spec["groups"], spec["softmax"], spec["positive_group"]
    g0, g1 = softmax_g[0], softmax_g[1]
    side_flags = spec["side_flags"]

    jimoks = {j.strip() for j in jimok.split(",") if j.strip()}
    meta = load_index(chips_dir / "index.csv")
    files = sorted(p for p in chips_dir.glob("*.png") if not p.name.startswith("_"))
    files = [f for f in files if meta.get(f.name, {}).get("jimok") in jimoks]
    if not files:
        raise SystemExit(f"[classify] 지목 {sorted(jimoks)} 칩 없음: {chips_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.get("cpu") else "cpu")
    print(f"[classify] {arch} | {device} | encoder={encoder} fill={fill} chip={chip_mode} "
          f"| 지목 '{jimok}' 칩 {len(files)} | prompt={Path(cfg['prompt_file']).name}")
    model, preprocess, tokenizer = load_model(device, arch)
    grid = ARCH_GRID.get(arch, 7)
    logit_scale = model.logit_scale.exp().item()

    # 그룹/서브 임베딩
    E_g0 = subprompt_embeddings(model, tokenizer, device, groups[g0])   # [n0, D]
    E_g1 = subprompt_embeddings(model, tokenizer, device, groups[g1])   # [n1, D]
    E_soft = torch.stack([E_g0.mean(0), E_g1.mean(0)])
    E_soft = E_soft / E_soft.norm(dim=-1, keepdim=True)
    E_side = {sf: subprompt_embeddings(model, tokenizer, device, groups[sf]).mean(0)
              for sf in side_flags if sf in groups}
    for k in E_side:
        E_side[k] = E_side[k] / E_side[k].norm()

    pos_is_g0 = (pos_g == g0)
    # dense 패치용 그룹 평균 (forest = 정상, changed = positive_group)
    E_forest_mean = E_soft[1] if pos_is_g0 else E_soft[0]
    E_changed_mean = E_soft[0] if pos_is_g0 else E_soft[1]

    rows, done = [], 0
    for grp in _chunks(files, batch):
        if encoder in ("whole", "cls"):
            ie = encode_whole(model, preprocess, device, grp, fill=fill)
        elif encoder in ("interior", "a1"):
            ie = encode_interior(model, preprocess, device, grp, grid,
                                 keep_tau=keep_tau, min_keep=min_keep,
                                 composite_black=(chip_mode != "real"))
        else:
            raise SystemExit(f"[classify] 알 수 없는 encoder: {encoder} (whole | interior)")

        P = (logit_scale * ie @ E_soft.T).softmax(-1).cpu().numpy()   # [b,2]  col0=g0 col1=g1
        S0 = (ie @ E_g0.T).cpu().numpy()
        S1 = (ie @ E_g1.T).cpu().numpy()
        side = {sf: (ie @ E_side[sf]).cpu().numpy() for sf in E_side}
        # 패치별 형질변경 비율 (부분 전환 포착 — decide 의 2차 트리거 + severity 신호 C)
        dp = dense_patch_change(model, preprocess, device, grp, grid,
                                E_forest_mean, E_changed_mean, logit_scale,
                                keep_tau=keep_tau, min_keep=min_keep, patch_thr=patch_thr,
                                last_mlp=last_mlp, composite_black=(chip_mode != "real"))

        for k, f in enumerate(grp):
            m = meta.get(f.name, {})
            p_g0, p_g1 = float(P[k, 0]), float(P[k, 1])
            p_forest = p_g0                                   # softmax[0] = 정상 쪽
            p_screen = p_g1 if pos_g == g1 else p_g0          # positive_group 확률
            # closest_pos = 정상(비-positive) 쪽, closest_neg = 형질변경(positive) 쪽
            forest_S, changed_S = (S0, S1) if not pos_is_g0 else (S1, S0)
            forest_names = groups[g0] if not pos_is_g0 else groups[g1]
            changed_names = groups[g1] if not pos_is_g0 else groups[g0]
            row = {
                "chip": f.name, "pnu": m.get("pnu", ""), "jibun": m.get("jibun", ""),
                "jimok": m.get("jimok", ""), "area_m2": m.get("area_m2", ""),
                "valid_ratio": m.get("valid_ratio", ""),
                "p_forest": round(p_forest, 4), "p_screen": round(p_screen, 4),
                "closest_pos": forest_names[int(forest_S[k].argmax())],
                "closest_pos_cos": round(float(forest_S[k].max()), 4),
                "closest_neg": changed_names[int(changed_S[k].argmax())],
                "closest_neg_cos": round(float(changed_S[k].max()), 4),
                "change_frac": dp[k][0], "change_frac50": dp[k][1], "n_patch": dp[k][2],
                # 서브프롬프트별 코사인 (선형프로브 피처용)
                "sims_normal": " ".join(f"{v:.4f}" for v in forest_S[k]),
                "sims_changed": " ".join(f"{v:.4f}" for v in changed_S[k]),
            }
            if side:
                sf = list(side)[0]
                row["gh_sim"] = round(float(side[sf][k]), 4)
                row["gh_dominant"] = sf
            rows.append(row)
        done += len(grp)
        print(f"  ...{done}/{len(files)}", end="\r")
    print()

    out = run_dir / "2_scores.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[classify] → {out}  ({len(rows)}행)")
    return out
