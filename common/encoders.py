#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이미지 → RemoteCLIP 임베딩. "이미지 처리 방식" 레버 두 개를 한 곳에 둔다.

  - encode_whole    : 전체 이미지 인코딩 (whole-image CLS). 폴리곤 밖 채움(fill) 선택
                      (mask 칩 전용 효과). docs/experiments.md §6d 에서 mean-fill 이 검정 대비 +0.036 F1.
                      실험 스크립트: classify_wholeimage.py ('v1')
  - encode_interior : 필지 내부만 인코딩. 폴리곤 밖 패치토큰 드롭 후 CLS. docs/experiments.md §3·§6c.
                      실험 스크립트: classify_interior.py ('A1')

`02_experiments/image_method/*` 와 `03_pipeline/stages/classify.py` 가 공유한다.
텍스트/프롬프트/임계값은 여기 없다 — 호출부가 갖는다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common.remoteclip_backbone import patch_coverage, encode_kept, chip_inputs

FILLS = ("black", "blackout", "mean", "gray", "inpaint")


def load_rgb(path: str | Path, fill: str = "black") -> Image.Image:
    """RGBA 칩(alpha = 폴리곤 마스크)을 RGB 로. 폴리곤 밖 채움 방식:
      black   : RGB 채널을 그대로 (mask 칩은 이미 검정이 구워져 있음 / real 칩은 원본 유지)
      blackout: alpha 로 폴리곤 밖을 (0,0,0) 으로 강제 합성 (real 칩 지오메트리 + 검정 바깥)
      mean    : 폴리곤 내부 픽셀 평균색
      gray    : ImageNet 평균 회색 (124,116,104)
      inpaint : cv2 TELEA 로 내부 텍스처 연장
    """
    im = Image.open(path)
    if fill == "black" or im.mode != "RGBA":
        return im.convert("RGB")
    if fill not in FILLS:
        raise SystemExit(f"알 수 없는 fill: {fill} (택1: {FILLS})")
    arr = np.array(im)
    rgb = arr[..., :3].astype(np.float32)
    inside = arr[..., 3] >= 128
    if not inside.any() or inside.all():
        return im.convert("RGB")
    if fill == "blackout":
        rgb[~inside] = 0.0
    elif fill == "mean":
        rgb[~inside] = rgb[inside].mean(0)
    elif fill == "gray":
        rgb[~inside] = np.array([124.0, 116.0, 104.0])
    elif fill == "inpaint":
        import cv2
        src = arr[..., :3].astype(np.uint8).copy()
        src[~inside] = 0
        rgb = cv2.inpaint(src, (~inside).astype(np.uint8) * 255, 3,
                          cv2.INPAINT_TELEA).astype(np.float32)
    return Image.fromarray(rgb.clip(0, 255).astype("uint8"), "RGB")


def _alpha01(path: str | Path) -> np.ndarray:
    a = np.array(Image.open(path).convert("RGBA").split()[-1])
    return (a > 127).astype(np.float32)


@torch.no_grad()
def encode_whole(model, preprocess, device, paths, fill: str = "black") -> torch.Tensor:
    """전체 이미지 인코딩. paths(list[Path]) → L2 정규화 이미지 임베딩 [N, D]. 한 배치로 처리."""
    px = torch.stack([preprocess(load_rgb(p, fill)) for p in paths]).to(device)
    ie = model.encode_image(px).float()
    return ie / ie.norm(dim=-1, keepdim=True)


@torch.no_grad()
def _dense_patch_vecs(v, px, keep_idx, keep_residual: bool = True, last_mlp: bool = False):
    """px[1,3,224,224], keep_idx=유지 패치(0..grid^2-1). 반환 내부 패치별 [K,D] (L2 정규화).
    마지막 transformer 블록에서 q·k attention 제거(MaskCLIP surgery) — 패치별 국소 특징 보존.
    02_experiments/image_method/classify_densepatch.dense_patch_vecs 와 동일."""
    x = v._embeds(px)
    sel = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device), keep_idx + 1])
    x = x.index_select(1, sel)
    blocks = v.transformer.resblocks
    for blk in blocks[:-1]:
        x = blk(x)
    last = blocks[-1]
    h = last.ln_1(x)
    W, b = last.attn.in_proj_weight, last.attn.in_proj_bias
    D = last.attn.embed_dim
    val = torch.nn.functional.linear(h, W[2 * D:3 * D, :], b[2 * D:3 * D])
    attn_out = last.attn.out_proj(val)
    x = (x + attn_out) if keep_residual else attn_out
    if last_mlp:
        x = x + last.mlp(last.ln_2(x))
    feat = v.ln_post(x)
    pv = feat[:, 1:, :] @ v.proj
    return (pv / pv.norm(dim=-1, keepdim=True))[0]


@torch.no_grad()
def dense_patch_change(model, preprocess, device, paths, grid: int,
                       E_forest: torch.Tensor, E_changed: torch.Tensor, logit_scale: float,
                       keep_tau: float = 0.5, min_keep: int = 4, patch_thr: float = 0.5,
                       last_mlp: bool = False, composite_black: bool = True) -> list[tuple]:
    """필지 내부 패치별 p_change(= changed 그룹 2-way softmax) → 필지당 집계.
    반환 paths 순서대로 [(change_frac, change_frac50, n_patch), ...]
      change_frac   = 커버리지 가중 평균 p_change     (0..1, "필지 내부 중 얼마나 바뀌어 보이나")
      change_frac50 = p_change > patch_thr 패치 커버리지 가중 비율
    E_forest / E_changed 는 L2 정규화된 그룹 평균 임베딩 [D]."""
    v = model.visual
    out = []
    for p in paths:
        px, alpha = chip_inputs(p, preprocess, device, composite_black=composite_black)
        cov = patch_coverage(alpha, grid)
        keep = np.where(cov >= keep_tau)[0]
        if len(keep) < min_keep:
            keep = np.argsort(-cov)[:min_keep]
        keep = np.sort(keep)
        w = cov[keep].astype(np.float64)
        if w.sum() < 0.5:
            w = np.ones_like(w)
        pv = _dense_patch_vecs(v, px, torch.tensor(keep, dtype=torch.long, device=device),
                               last_mlp=last_mlp)                       # [K,D]
        sf = (pv @ E_forest).cpu().numpy()
        sc = (pv @ E_changed).cpu().numpy()
        logits = np.stack([logit_scale * sf, logit_scale * sc], 1)
        logits -= logits.max(1, keepdims=True)
        e = np.exp(logits)
        p_change = e[:, 1] / e.sum(1)                                   # [K]
        W = w.sum()
        out.append((round(float((w * p_change).sum() / W), 4),
                    round(float((w * (p_change > patch_thr)).sum() / W), 4),
                    int(len(keep))))
    return out


@torch.no_grad()
def encode_interior(model, preprocess, device, paths, grid: int,
                    keep_tau: float = 0.5, min_keep: int = 4, composite_black: bool = True) -> torch.Tensor:
    """필지 내부만 인코딩 — 폴리곤 밖 패치토큰 드롭. paths 를 하나씩 처리 → [N, D].
    composite_black=False 는 real 칩(밖도 실영상, alpha 는 패치선택에만)."""
    visual = model.visual
    out = []
    for p in paths:
        im = Image.open(p).convert("RGBA")
        alpha = _alpha01(p)
        if composite_black:
            rgb = Image.alpha_composite(Image.new("RGBA", im.size, (0, 0, 0, 255)), im).convert("RGB")
        else:
            rgb = im.convert("RGB")
        px = preprocess(rgb).unsqueeze(0).to(device)
        cov = patch_coverage(alpha, grid)
        keep = np.where(cov >= keep_tau)[0]
        if len(keep) < min_keep:
            keep = np.argsort(-cov)[:min_keep]
        keep_idx = torch.tensor(np.sort(keep), dtype=torch.long, device=device)
        out.append(encode_kept(visual, px, keep_idx))
    return torch.cat(out, 0)
