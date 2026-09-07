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

from common.remoteclip_backbone import patch_coverage, encode_kept

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
