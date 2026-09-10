#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
저장소 표준 경로 — 어느 폴더에서 실행하든 저장소 루트 기준 절대경로로 해결된다.

    from common.paths import ORTHO_GEOREF, PARCELS, CHIPS
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- 01_preprocess -----------------------------------------------------------
PRE = ROOT / "01_preprocess"
RAW = PRE / "raw"
RAW_SATELLITE = RAW / "satellite"
RAW_CADASTRAL = RAW / "cadastral"
DERIVED = PRE / "derived"
CHIPS = PRE / "chips"
REFERENCE = PRE / "reference"
LABELS = PRE / "labels"

# 파일럿 도엽 36813045 (경북 김천시, 동부원점 EPSG:5187) 의 표준 산출물
SHEET = "36813045"
ORTHO_RAW = RAW_SATELLITE / "(B060)정사영상_2025_36813045.tif"
ORTHO_META_XML = RAW_SATELLITE / "(B060)정사영상메타데이터_202512000536813045.xml"
ORTHO_GEOREF = DERIVED / "(B060)정사영상_2025_36813045_georef.tif"
PARCELS = DERIVED / "gimcheon_36813045_parcels_within.gpkg"
CADASTRE_SHP = RAW_CADASTRAL / "LSMD_CONT_LDREG_경북_김천시" / "LSMD_CONT_LDREG_47150_202608.shp"

LABEL_SET = LABELS / "imya_eval_150"        # 초기 임야 zero-shot 평가용 150 (면적층화)
IMYA_EVAL_SET = LABELS / "imya_eval"        # 손라벨 전량 검증셋 (임야 vs 형질변경, 7도엽)
NONGJI_EVAL_SET = LABELS / "nongji_eval"    # 손라벨 전량 검증셋 (농지 vs 형질변경, 6도엽)
JIMOK_TAXONOMY = REFERENCE / "jimok_taxonomy.json"
JIMOK_CODES = REFERENCE / "jimok_codes.csv"


def chip_dir(mode: str, sheet: str = SHEET, group: str | None = None) -> Path:
    """s3_make_chips.py 산출 폴더 규칙:
      group 지정  01_preprocess/chips/<sheet>_<group>_<mode>/   (예: 37714092_imya_real)
      미지정      01_preprocess/chips/<sheet>_<mode>/
    """
    tag = f"{sheet}_{group}_{mode}" if group else f"{sheet}_{mode}"
    return CHIPS / tag


# --- 02_experiments --------------------------------------------------------
EXPERIMENTS = ROOT / "02_experiments"
EXPERIMENT_RESULTS = EXPERIMENTS / "results"
PROMPT_ABLATION = EXPERIMENTS / "prompt_ablation"
PROMPT_SETS_DIR = PROMPT_ABLATION / "sets"              # 사용자가 편집하는 프롬프트 세트 <name>.json
PROMPT_CANDIDATES_DIR = PROMPT_ABLATION / "candidates"  # probe 용 문구 후보 풀
PROMPT_WORK_SUBSET = PROMPT_ABLATION / "_work_subset.csv"  # 동결된 개발용 부분집합 (pnu,split)

# --- 03_pipeline ---------------------------------------------------------
PIPELINE = ROOT / "03_pipeline"
PIPELINE_CONFIGS = PIPELINE / "configs"
PIPELINE_PROMPTS = PIPELINE / "prompts"
PIPELINE_RUNS = PIPELINE / "runs"

# --- models ----------------------------------------------------------------
MODELS = ROOT / "models"
