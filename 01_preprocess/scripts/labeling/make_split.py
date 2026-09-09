#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
손라벨 150 → dev/test 50:50 층화 분할. 1회용. 실행 후 커밋하고 다시 만들지 않는다.

층화: 면적 5분위(labels.csv 의 area_m2 로 계산) × label. 각 (분위, label) 셀 안에서
셔플 후 절반씩 dev/test. 홀수 셀은 dev 쪽에 +1 (셀 순서 결정적).

출력: data/labels/imya_eval_150/split.csv  (pnu,jibun,area_m2,label,area_q,split)

사용:
    python scripts/labeling/make_split.py
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.paths import LABEL_SET  # noqa: E402

OUT_DIR = LABEL_SET
LABELS = OUT_DIR / "labels.csv"
SPLIT = OUT_DIR / "split.csv"
SEED = 42
NQ = 5


def quintile_edges(vals: list[float], nq: int) -> list[float]:
    """nq-분위 상단 경계 (nq-1 개). 선형보간."""
    s = sorted(vals)
    n = len(s)
    edges = []
    for k in range(1, nq):
        pos = k / nq * (n - 1)
        lo = int(pos)
        frac = pos - lo
        hi = min(lo + 1, n - 1)
        edges.append(s[lo] + frac * (s[hi] - s[lo]))
    return edges


def q_of(v: float, edges: list[float]) -> int:
    for i, e in enumerate(edges):
        if v <= e:
            return i
    return len(edges)


def main() -> None:
    with open(LABELS, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    # finalize_labels.py 가 이미 '제외' 를 뺐지만, 혹시 섞여 들어와도 방어
    rows = [r for r in rows if r["label"] in ("형질변경", "임야", "보류")]
    for r in rows:
        r["area_m2"] = float(r["area_m2"])
    rows.sort(key=lambda r: r["pnu"])  # 결정적 시작 순서

    edges = quintile_edges([r["area_m2"] for r in rows], NQ)
    for r in rows:
        r["area_q"] = q_of(r["area_m2"], edges)

    rng = random.Random(SEED)
    assign: dict[str, str] = {}
    # 셀 순서 결정적: (area_q, label) 정렬. 홀수 셀의 잉여 1개는 dev/test 번갈아.
    cells: dict[tuple, list] = {}
    for r in rows:
        cells.setdefault((r["area_q"], r["label"]), []).append(r)
    odd_to_dev = True
    for key in sorted(cells):
        cell = cells[key]
        idx = list(range(len(cell)))
        rng.shuffle(idx)
        if len(cell) % 2:
            n_dev = len(cell) // 2 + (1 if odd_to_dev else 0)
            odd_to_dev = not odd_to_dev
        else:
            n_dev = len(cell) // 2
        for j, i in enumerate(idx):
            assign[cell[i]["pnu"]] = "dev" if j < n_dev else "test"

    rows.sort(key=lambda r: r["pnu"])
    with open(SPLIT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["pnu", "jibun", "area_m2", "label", "area_q", "split"])
        for r in rows:
            w.writerow([r["pnu"], r["jibun"], r["area_m2"], r["label"],
                        r["area_q"], assign[r["pnu"]]])

    # 요약
    from collections import Counter
    cnt = Counter((assign[r["pnu"]], r["label"]) for r in rows)
    qcnt = Counter((assign[r["pnu"]], r["area_q"]) for r in rows)
    print(f"→ {SPLIT}  ({len(rows)}개)  seed={SEED}")
    print(f"면적 분위 경계(㎡): {[round(e, 1) for e in edges]}")
    for sp in ("dev", "test"):
        tot = sum(v for (s, _), v in cnt.items() if s == sp)
        byl = " · ".join(f"{lab} {cnt[(sp, lab)]}" for lab in ("형질변경", "임야"))
        byq = " ".join(f"q{q}:{qcnt[(sp, q)]}" for q in range(NQ))
        print(f"  {sp:<4} n={tot}  {byl}   [{byq}]")


if __name__ == "__main__":
    main()
