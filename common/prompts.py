#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
프롬프트 세트 로딩 — 실험(2그룹 pos/neg)과 파이프라인(N그룹) 공용.

파일 형식 두 가지를 모두 받는다:

  (A) 레거시 2그룹        {"positive": [...], "negative": [...]}
  (B) N그룹               {"groups": {"forest": [...], "notforest": [...]},
                          "softmax": ["forest", "notforest"],
                          "positive_group": "notforest",
                          "side_flags": []}          # 선택

(A) 는 로드 시 (B) 로 정규화된다: groups=forest/notforest, softmax=[forest,notforest],
positive_group=notforest.
"""
from __future__ import annotations

import json
from pathlib import Path

# 임야 whole-image CLS 실험용 인라인 세트 (docs/experiments.md §1·§5)
IMYA_V1_POS = [
    "a forest",
    "dense green woodland",
    "a tree-covered hill",
    "a forested mountain slope",
    "a wooded area with many trees",
    "leafless deciduous forest in winter",
    "bare brown trees covering a hillside",
]
IMYA_V1_NEG = [
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
IMYA_V2_POS = [
    "a hillside densely covered with tree canopy",
    "a mix of evergreen and leafless deciduous trees on rough terrain",
    "an uncultivated wooded slope with no roads or buildings",
    "forest with irregular natural tree spacing, not planted in rows",
    "a bare brown deciduous forest canopy in winter",
    "a steep forested mountainside",
]
IMYA_V2_NEG = [
    "a patch of bare soil with tree stumps where forest was cleared",
    "graded flat bare earth with tire tracks cut into a wooded hill",
    "a metal-roofed shed or building surrounded by trees",
    "rows of dark solar panels on a cleared hillside",
    "a new dirt or paved road cut through forest",
    "grave mounds on a cleared hillside",
    "a cultivated field or greenhouse on former forest land",
]

# 실험 스크립트가 --prompts {v1,v2} 로 고르는 2그룹 세트
PROMPT_SETS = {"v1": (IMYA_V1_POS, IMYA_V1_NEG), "v2": (IMYA_V2_POS, IMYA_V2_NEG)}


def load_prompt_pair(path: str | Path) -> tuple[list[str], list[str]]:
    """실험 스크립트용 2그룹 반환 = (임야다움 문구, 형질변경다움 문구).

    실험 코드 관례상 첫 번째가 POSITIVE(임야), 두 번째가 NEGATIVE(형질변경).
    프롬프트 파일의 `positive_group` 은 '형질변경 의심' 그룹을 뜻하므로 여기서 뒤집는다.
    """
    spec = load_groups(path)
    a, b = spec["softmax"][0], spec["softmax"][1]
    changed_grp = spec["positive_group"]              # 형질변경다움
    forest_grp = b if changed_grp == a else a         # 임야다움
    g = spec["groups"]
    return list(g[forest_grp]), list(g[changed_grp])


def load_groups(path: str | Path) -> dict:
    """정규화된 N그룹 스펙 dict 반환."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "groups" in d:
        groups = {k: list(v) for k, v in d["groups"].items()}
        softmax = list(d.get("softmax", list(groups)[:2]))
        positive_group = d.get("positive_group", softmax[-1])
        side_flags = list(d.get("side_flags", []))
    elif "positive" in d and "negative" in d:
        groups = {"forest": list(d["positive"]), "notforest": list(d["negative"])}
        softmax = ["forest", "notforest"]
        positive_group = "notforest"
        side_flags = []
    else:
        raise SystemExit(f"프롬프트 파일 형식 불명 (positive/negative 또는 groups 필요): {path}")
    for g in softmax:
        if not groups.get(g):
            raise SystemExit(f"softmax 그룹 '{g}' 이 비었음: {path}")
    return {"groups": groups, "softmax": softmax,
            "positive_group": positive_group, "side_flags": side_flags}
