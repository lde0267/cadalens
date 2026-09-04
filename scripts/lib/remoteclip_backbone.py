#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RemoteCLIP 공용 백본 유틸 — 임야/농지 파이프라인이 함께 쓴다.

담는 것:
  - 모델 로드 (HuggingFace chendelong/RemoteCLIP ViT-B-32, models/remoteclip/ 캐시)
  - 텍스트 프롬프트 -> 그룹 임베딩
  - 칩 로드 (RGBA -> 검정합성 RGB + alpha 마스크)
  - 폴리곤 커버리지 기반 내부 패치 선택
  - A1 방식 토큰 드롭 이미지 인코딩 (폴리곤 밖 패치토큰 제거)

여기에는 '무엇을 분류할지'(프롬프트, 임계값, 산출물)는 넣지 않는다.
그건 scripts/imya/*, scripts/nongji/* 각자가 갖는다.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import open_clip
from huggingface_hub import hf_hub_download

CKPT_DIR = Path("models/remoteclip")
HF_REPO = "chendelong/RemoteCLIP"
ARCH = "ViT-B-32"          # 기본
CKPT_FILE = "RemoteCLIP-ViT-B-32.pt"

ARCH_CKPT = {
    "ViT-B-32": "RemoteCLIP-ViT-B-32.pt",
    "ViT-L-14": "RemoteCLIP-ViT-L-14.pt",
    "RN50": "RemoteCLIP-RN50.pt",
}
ARCH_GRID = {"ViT-B-32": 7, "ViT-L-14": 16}   # 224 / patch  (7x7=49  vs  16x16=256)

PATCH = 32          # ViT-B-32 patch size (하위호환)
GRID = 7            # 224 / 32

TEMPLATES = [
    "a satellite photo of {}.",
    "an aerial photo of {}.",
    "a high-resolution aerial image of {}.",
    "a remote sensing image of {}.",
]


# ---------------------------------------------------------------- 모델
def resolve_ckpt(arch: str = ARCH) -> str:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    fn = ARCH_CKPT[arch]
    print(f"  가중치: {HF_REPO}/{fn}")
    return hf_hub_download(HF_REPO, fn, cache_dir=str(CKPT_DIR))


def load_model(device: torch.device, arch: str = ARCH):
    """반환: (model, preprocess, tokenizer). model 은 eval, device 로 이동됨."""
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
    return model, preprocess, tokenizer


# ---------------------------------------------------------------- 텍스트
@torch.no_grad()
def subprompt_embeddings(model, tokenizer, device, phrases, templates=TEMPLATES):
    """각 문구 -> (문구 x 템플릿) 평균, L2 정규화. 반환 (N, D)."""
    embs = []
    for p in phrases:
        prompts = [t.format(p) for t in templates]
        e = model.encode_text(tokenizer(prompts).to(device)).float()
        e = e / e.norm(dim=-1, keepdim=True)
        embs.append(e.mean(0))
    E = torch.stack(embs)
    return E / E.norm(dim=-1, keepdim=True)


@torch.no_grad()
def group_embeddings(model, tokenizer, device, groups: dict[str, list[str]]):
    """{그룹명: [문구...]} -> {그룹명: 정규화된 그룹평균벡터}, 그리고 (하위문구 리스트, 하위임베딩)."""
    names, subs, sub_names = [], [], []
    per_group = {}
    for g, phrases in groups.items():
        E = subprompt_embeddings(model, tokenizer, device, phrases)
        gv = E.mean(0)
        per_group[g] = gv / gv.norm()
        names.append(g)
        subs.append(E)
        sub_names += [(g, p) for p in phrases]
    return per_group, sub_names, torch.cat(subs, 0)


# ---------------------------------------------------------------- 칩 / 패치
def load_index(p: Path) -> dict:
    if not Path(p).exists():
        return {}
    with open(p, encoding="utf-8-sig") as f:
        return {r["chip"]: r for r in csv.DictReader(f)}


def chip_inputs(path: Path, preprocess, device, composite_black: bool = True):
    """
    RGBA 칩 -> (px[1,3,224,224], alpha01[224,224]).
    composite_black=True  : 폴리곤 밖을 검정으로 합성  (mask 칩)
    composite_black=False : alpha 무시하고 원본 RGB 유지 (real 칩 - 밖도 실제 영상)
    alpha01 은 두 경우 모두 폴리곤 마스크(패치 선택용).
    """
    im = Image.open(path).convert("RGBA")
    alpha = (np.array(im.split()[-1]) > 127).astype(np.float32)
    if composite_black:
        rgb = Image.alpha_composite(Image.new("RGBA", im.size, (0, 0, 0, 255)), im).convert("RGB")
    else:
        rgb = im.convert("RGB")
    px = preprocess(rgb).unsqueeze(0).to(device)
    return px, alpha


def patch_coverage(alpha01: np.ndarray, grid: int = GRID) -> np.ndarray:
    """alpha (S,S) in {0,1} -> (grid*grid,) 행-우선 패치별 내부 비율 (open_clip 토큰 순서)."""
    s = alpha01.shape[0]
    p = s // grid
    a = alpha01[:grid * p, :grid * p].reshape(grid, p, grid, p).mean(axis=(1, 3))
    return a.reshape(-1)


def interior_patches(alpha01: np.ndarray, keep_tau: float, min_keep: int, grid: int = GRID):
    """반환: (keep_idx np.int, weights np.float). 유지 패치가 min_keep 미만이면 커버리지 상위로 채움."""
    cov = patch_coverage(alpha01, grid)
    keep = np.where(cov >= keep_tau)[0]
    if len(keep) < min_keep:
        keep = np.argsort(-cov)[:min_keep]
    keep = np.sort(keep)
    w = cov[keep].astype(np.float64)
    if w.sum() < 0.5:
        w = np.ones_like(w)
    return keep, w


@torch.no_grad()
def encode_kept(visual, px: torch.Tensor, keep_idx: torch.Tensor) -> torch.Tensor:
    """
    A1 인코딩 — _embeds 후 (CLS + 유지 패치토큰)만 transformer 통과.
    폴리곤 밖 패치를 시퀀스에서 제거 = 전 블록에서 외부 key 마스킹과 등가.
    반환: L2 정규화된 이미지 임베딩 [1, D].
    """
    x = visual._embeds(px)                                  # [1, 1+GRID^2, W]  pos_emb·ln_pre 완료
    sel = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device), keep_idx + 1])
    x = x.index_select(1, sel)                              # [1, 1+K, W]
    x = visual.transformer(x)
    x = visual.ln_post(x)
    pooled = x[:, 0]
    if visual.proj is not None:
        pooled = pooled @ visual.proj
    return pooled / pooled.norm(dim=-1, keepdim=True)
