#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
프롬프트 문구 선택 파이프라인 (A1~A4).

  A1/A2  후보 문구를 각각 단일 프로토타입으로 놓고 반대 그룹은 풀 전체 평균 → dev AP(PR-AUC) 순위
  A3     AP 순위대로 greedy 전진선택 (개선 게이트 +0.01 AP, 그룹당 상한 8)
  A4     greedy 결과에 LOO — 빼도 AP 안 떨어지는 문구 제거. references 세트도 LOO 감사
  최종   {greedy, greedy+LOO, v1, e1, v3} 를 고정 split test 로 채점 + 5회 랜덤 재분할 오차막대

입력: 02_experiments/prompt_ablation/{pool_positive,pool_negative,references}.json
이미지: 01_preprocess/chips/37714092_real_nopad + fill=mean + whole 인코딩 (최고 구성)
라벨: 01_preprocess/labels/imya_eval_150/{labels,split}.csv   (positive = 형질변경)

사용: python 02_experiments/prompt_ablation/prompt_select.py [--cpu]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import LABEL_SET, chip_dir, EXPERIMENT_RESULTS  # noqa: E402
from common.remoteclip_backbone import load_model, subprompt_embeddings, load_index  # noqa: E402
from common.encoders import encode_whole  # noqa: E402
from common.scoring import read_csv  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = EXPERIMENT_RESULTS / "prompt_select"
POS_LABEL = "형질변경"
GATE = 0.01          # A3 개선 게이트 (dev AP)
LOO_EPS = 0.003      # A4: 빼도 AP 가 이만큼도 안 떨어지면 제거
CAP = 8              # 그룹당 최대 문구 수


# ---------------------------------------------------------------- 데이터
def load_pool(name: str) -> list[str]:
    d = json.loads((HERE / name).read_text(encoding="utf-8"))
    return [c["text"] for c in d["candidates"]]


def load_refs() -> dict[str, dict]:
    d = json.loads((HERE / "references.json").read_text(encoding="utf-8"))
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ---------------------------------------------------------------- 프로토타입/스코어
class Scorer:
    """텍스트 임베딩·이미지 임베딩을 미리 계산해두고, 문구 리스트만 바꿔가며 즉시 채점."""

    def __init__(self, model, tokenizer, device, phrases: list[str], img_emb: np.ndarray):
        E = subprompt_embeddings(model, tokenizer, device, phrases).cpu().numpy()  # [P, D] L2정규화
        self.emb = {p: E[i] for i, p in enumerate(phrases)}
        self.img = img_emb                                                          # [N, D] L2정규화
        self.logit_scale = float(model.logit_scale.exp().item())

    def proto(self, phrases: list[str]) -> np.ndarray:
        v = np.mean([self.emb[p] for p in phrases], axis=0)
        return v / (np.linalg.norm(v) + 1e-9)

    def p_forest(self, forest: list[str], changed: list[str]) -> np.ndarray:
        Ef, Ec = self.proto(forest), self.proto(changed)
        lf = self.logit_scale * (self.img @ Ef)
        lc = self.logit_scale * (self.img @ Ec)
        m = np.maximum(lf, lc)
        ef, ec = np.exp(lf - m), np.exp(lc - m)
        return ef / (ef + ec)                                                       # [N]


def ap(p_forest: np.ndarray, y_changed: np.ndarray) -> float:
    """positive = 형질변경. 스코어 = 1 - p_forest."""
    return float(average_precision_score(y_changed, 1.0 - p_forest))


def auc(p_forest: np.ndarray, y_changed: np.ndarray) -> float:
    return float(roc_auc_score(y_changed, 1.0 - p_forest))


def f1_at_dev_tau(p_dev, y_dev, p_test, y_test):
    taus = np.round(np.arange(0.50, 0.996, 0.005), 3)
    best_t, best_f1 = 0.5, -1.0
    for t in taus:
        pred = p_dev < t
        tp = int((pred & (y_dev == 1)).sum()); fp = int((pred & (y_dev == 0)).sum())
        fn = int((~pred & (y_dev == 1)).sum())
        f1 = tp / (tp + 0.5 * (fp + fn)) if tp else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    pred = p_test < best_t
    tp = int((pred & (y_test == 1)).sum()); fp = int((pred & (y_test == 0)).sum())
    fn = int((~pred & (y_test == 1)).sum()); tn = int((~pred & (y_test == 0)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    band = float(np.mean(np.abs(p_test - best_t) <= 0.05))
    return best_t, P, R, F1, band


# ---------------------------------------------------------------- A3 greedy
def greedy(sc: Scorer, pos_pool, neg_pool, y, dev, rank_pos, rank_neg, log):
    forest = [rank_pos[0]]
    changed = [rank_neg[0]]
    cur = ap(sc.p_forest(forest, changed)[dev], y[dev])
    log.append({"step": 0, "add": f"seed forest={forest[0]!r} changed={changed[0]!r}", "dev_AP": round(cur, 4)})
    rem = [("f", p) for p in pos_pool if p not in forest] + [("c", n) for n in neg_pool if n not in changed]
    step = 1
    while rem:
        best = None
        for grp, cand in rem:
            if grp == "f" and len(forest) >= CAP:
                continue
            if grp == "c" and len(changed) >= CAP:
                continue
            f2 = forest + [cand] if grp == "f" else forest
            c2 = changed + [cand] if grp == "c" else changed
            a = ap(sc.p_forest(f2, c2)[dev], y[dev])
            if best is None or a > best[0]:
                best = (a, grp, cand)
        if best is None or best[0] - cur < GATE:
            log.append({"step": step, "add": "STOP (개선 < gate)", "dev_AP": round(cur, 4),
                        "best_reject": round(best[0], 4) if best else None})
            break
        a, grp, cand = best
        (forest if grp == "f" else changed).append(cand)
        rem = [(g, p) for g, p in rem if p != cand]
        cur = a
        log.append({"step": step, "add": f"{'forest' if grp == 'f' else 'changed'} += {cand!r}",
                    "dev_AP": round(cur, 4)})
        step += 1
    return forest, changed, cur


# ---------------------------------------------------------------- A4 LOO
def loo_prune(sc: Scorer, forest, changed, y, dev, log, tag="greedy"):
    forest, changed = list(forest), list(changed)
    changed_any = True
    while changed_any:
        changed_any = False
        cur = ap(sc.p_forest(forest, changed)[dev], y[dev])
        for grp_name, grp in (("forest", forest), ("changed", changed)):
            if len(grp) <= 1:
                continue
            for ph in list(grp):
                trial = [x for x in grp if x != ph]
                f2 = trial if grp_name == "forest" else forest
                c2 = trial if grp_name == "changed" else changed
                a = ap(sc.p_forest(f2, c2)[dev], y[dev])
                if a >= cur - LOO_EPS:
                    grp.remove(ph)
                    log.append({"tag": tag, "drop": f"{grp_name} -= {ph!r}",
                                "dev_AP_before": round(cur, 4), "dev_AP_after": round(a, 4)})
                    changed_any = True
                    cur = ap(sc.p_forest(forest, changed)[dev], y[dev])
    return forest, changed


def loo_audit(sc: Scorer, forest, changed, y, dev):
    """references 세트: 빼도 AP 안 떨어지는(=기여 없는) 문구만 리포트."""
    cur = ap(sc.p_forest(forest, changed)[dev], y[dev])
    dead = []
    for grp_name, grp in (("forest", forest), ("changed", changed)):
        for ph in grp:
            trial = [x for x in grp if x != ph] or grp
            f2 = trial if grp_name == "forest" else forest
            c2 = trial if grp_name == "changed" else changed
            a = ap(sc.p_forest(f2, c2)[dev], y[dev])
            if a >= cur - LOO_EPS:
                dead.append({"grp": grp_name, "phrase": ph, "dev_AP_drop": round(cur - a, 4)})
    return round(cur, 4), dead


# ---------------------------------------------------------------- main
def main():
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--cpu", action="store_true")
    args = ap_.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    OUT.mkdir(parents=True, exist_ok=True)

    pos_pool = load_pool("pool_positive.json")
    neg_pool = load_pool("pool_negative.json")
    refs = load_refs()
    all_phrases = sorted(set(pos_pool) | set(neg_pool)
                         | {p for r in refs.values() for p in r["positive"] + r["negative"]})
    print(f"후보 pos {len(pos_pool)} / neg {len(neg_pool)} · 참조 {len(refs)} · 고유 문구 {len(all_phrases)}")

    # 라벨 + split
    labels = {r["pnu"]: r["label"] for r in read_csv(LABEL_SET / "labels.csv")}
    split = {r["pnu"]: r["split"] for r in read_csv(LABEL_SET / "split.csv")}
    cdir = chip_dir("real").parent / "37714092_real_nopad"
    meta = load_index(cdir / "index.csv")
    chip_by_pnu = {m["pnu"]: cdir / m["chip"] for m in meta.values()}
    rows = [(pnu, chip_by_pnu[pnu], lab, split.get(pnu, ""))
            for pnu, lab in labels.items() if lab != "보류" and pnu in chip_by_pnu]
    pnus = [r[0] for r in rows]
    paths = [r[1] for r in rows]
    y = np.array([1 if r[2] == POS_LABEL else 0 for r in rows])
    sp = np.array([r[3] for r in rows])
    dev = np.where(sp == "dev")[0]
    test = np.where(sp == "test")[0]
    print(f"라벨 필지 {len(rows)}  (dev {len(dev)} / test {len(test)}) · 형질변경 {int(y.sum())} / 임야 {int((y == 0).sum())}")

    # 모델 · 임베딩 (모델 호출 1회)
    model, preprocess, tokenizer = load_model(device, "ViT-B-32")
    with torch.no_grad():
        img_emb = encode_whole(model, preprocess, device, paths, fill="mean").cpu().numpy()
    sc = Scorer(model, tokenizer, device, all_phrases, img_emb)
    print("임베딩 완료. 이후는 numpy 연산.\n")

    # ---------- A1/A2 ----------
    neg_full = sc.proto(neg_pool)
    pos_full = sc.proto(pos_pool)
    rankrows = []
    for p in pos_pool:
        pf = _pf_raw(sc, sc.proto([p]), neg_full)
        rankrows.append(("forest", p, ap(pf[dev], y[dev]), auc(pf[dev], y[dev]), ap(pf[test], y[test])))
    for n in neg_pool:
        pf = _pf_raw(sc, pos_full, sc.proto([n]))
        rankrows.append(("changed", n, ap(pf[dev], y[dev]), auc(pf[dev], y[dev]), ap(pf[test], y[test])))
    rankrows.sort(key=lambda r: -r[2])
    with open(OUT / "A1_ranking.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group", "phrase", "dev_AP", "dev_AUC", "test_AP"])
        for g, ph, dap, dau, tap in rankrows:
            w.writerow([g, ph, round(dap, 4), round(dau, 4), round(tap, 4)])
    rank_pos = [ph for g, ph, *_ in rankrows if g == "forest"]
    rank_neg = [ph for g, ph, *_ in rankrows if g == "changed"]
    print("── A1/A2 상위 6 ──")
    for g, ph, dap, dau, tap in rankrows[:6]:
        print(f"  {g:7} devAP {dap:.3f} devAUC {dau:.3f} testAP {tap:.3f}  {ph}")

    # ---------- A3 greedy ----------
    glog = []
    g_forest, g_changed, g_devap = greedy(sc, pos_pool, neg_pool, y, dev, rank_pos, rank_neg, glog)
    (OUT / "A3_greedy.json").write_text(json.dumps(
        {"forest": g_forest, "changed": g_changed, "dev_AP": round(g_devap, 4), "trajectory": glog},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n── A3 greedy → forest {len(g_forest)} / changed {len(g_changed)} · dev AP {g_devap:.3f} ──")
    for e in glog:
        print("   ", e.get("add", e))

    # ---------- A4 LOO ----------
    llog = []
    l_forest, l_changed = loo_prune(sc, g_forest, g_changed, y, dev, llog)
    audits = {}
    for name, r in refs.items():
        cur, dead = loo_audit(sc, r["positive"], r["negative"], y, dev)
        audits[name] = {"dev_AP": cur, "dead_phrases": dead}
    (OUT / "A4_final.json").write_text(json.dumps(
        {"greedy_LOO": {"forest": l_forest, "changed": l_changed}, "loo_log": llog,
         "reference_audit": audits}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n── A4 LOO → forest {len(l_forest)} / changed {len(l_changed)} ──")
    for e in llog:
        print("   ", e["drop"], f"(devAP {e['dev_AP_before']}→{e['dev_AP_after']})")
    if not llog:
        print("    (제거할 문구 없음)")

    # ---------- 최종 채점 ----------
    contenders = {
        "greedy": (g_forest, g_changed),
        "greedy_LOO": (l_forest, l_changed),
        **{k: (v["positive"], v["negative"]) for k, v in refs.items()},
    }
    print("\n########## 최종 비교 (고정 split) ##########")
    print(f"{'세트':<14} {'nF/nC':>7}  devAP  testAP testAUC  τ*    P     R     F1     밴드")
    print("-" * 76)
    final = {}
    for name, (F, C) in contenders.items():
        pf = sc.p_forest(F, C)
        d_ap, t_ap, t_auc = ap(pf[dev], y[dev]), ap(pf[test], y[test]), auc(pf[test], y[test])
        tau, P, R, F1, band = f1_at_dev_tau(pf[dev], y[dev], pf[test], y[test])
        final[name] = dict(nF=len(F), nC=len(C), dev_AP=round(d_ap, 4), test_AP=round(t_ap, 4),
                           test_AUC=round(t_auc, 4), tau=tau, P=round(P, 3), R=round(R, 3),
                           F1=round(F1, 3), band=round(band, 3))
        print(f"{name:<14} {len(F):>3}/{len(C):<3}  {d_ap:.3f}  {t_ap:.3f}  {t_auc:.3f}  "
              f"{tau:.2f}  {P:.2f}  {R:.2f}  {F1:.3f}  {band:.0%}")

    # ---------- 5회 랜덤 재분할 오차막대 (최종셋 + 참조) ----------
    rng = np.random.default_rng(0)
    idx_all = np.arange(len(rows))
    boot = {}
    for name, (F, C) in contenders.items():
        pf = sc.p_forest(F, C)
        aps = []
        for s in range(5):
            perm = rng.permutation(idx_all)
            te = perm[:75]
            aps.append(ap(pf[te], y[te]))
        boot[name] = (float(np.mean(aps)), float(np.std(aps)))
    print("\n── 5회 랜덤 75 재분할 test AP (mean±std) ──")
    for name, (mu, sd) in boot.items():
        print(f"  {name:<14} {mu:.3f} ± {sd:.3f}")

    (OUT / "SUMMARY.json").write_text(json.dumps(
        {"final_fixed_split": final, "bootstrap_testAP": {k: {"mean": round(v[0], 4), "std": round(v[1], 4)}
                                                          for k, v in boot.items()},
         "selected": {"forest": l_forest, "changed": l_changed}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {OUT}/  (A1_ranking.csv · A3_greedy.json · A4_final.json · SUMMARY.json)")


def _pf_raw(sc: "Scorer", Ef: np.ndarray, Ec: np.ndarray) -> np.ndarray:
    lf = sc.logit_scale * (sc.img @ Ef)
    lc = sc.logit_scale * (sc.img @ Ec)
    m = np.maximum(lf, lc)
    ef, ec = np.exp(lf - m), np.exp(lc - m)
    return ef / (ef + ec)


if __name__ == "__main__":
    main()
