# -*- coding: utf-8 -*-
"""CadaLens 검증 성능 + RemoteCLIP 프롬프트 유사도 시각화.

    python 03_pipeline/make_figs.py            # 전체
    python 03_pipeline/make_figs.py clip       # fig4/fig5 (프롬프트 유사도)만

출력: 03_pipeline/runs/_figs/
"""
from __future__ import annotations
import csv, json, glob, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for cand in ("Malgun Gothic", "NanumGothic", "Yu Gothic"):
    if any(f.name == cand for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "03_pipeline" / "runs"
DATA = ROOT / "03_pipeline" / "data"
OUT = RUNS / "_figs"
OUT.mkdir(parents=True, exist_ok=True)

C_NORM = "#2e7d32"   # 정상(임야/농지=farm)
C_CHG = "#e65100"    # 형질변경(changed/built)
C_GH = "#6a1b9a"     # 비닐하우스
C_BEST = "#1565c0"
C_FIX = "#b0bec5"

SHEETS = ["35604085", "36709030", "36710022", "36813045", "37709030", "37714092", "37813048"]
SHEET_NAME = {"35604085": "익산085\n(김제)", "36709030": "공주030\n(세종금남)",
              "36710022": "공주022\n(세종)", "36813045": "김천045\n(경북김천)",
              "37709030": "수원030\n(경기광주)", "37714092": "안성092\n(경기안성)",
              "37813048": "제천048\n(충북제천)"}

# ── 프롬프트 문장 → 짧은 한글 표기 (fig4/fig5 막대 라벨용) ──────────────
SHORT = {
    # imya_change.json — forest
    "a hillside covered with dense green forest canopy": "울창한 수관",
    "a slope of mixed evergreen pine and bare brown winter oak trees": "침엽+겨울참나무",
    "a rough natural canopy of individual tree crowns following steep terrain": "급경사 자연림",
    "dark evergreen conifer forest along a mountain ridge": "능선 침엽수림",
    "a conifer plantation with young trees planted in regular rows on a mountainside": "침엽수 조림지",
    "a wooded mountain slope with only a narrow bare forest road or firebreak through it": "임도/방화선만",
    "a thin young forest of scattered small trees over rough grass and shrubs on a slope": "성긴 어린 숲",
    "natural woodland with irregular tree spacing and no straight cleared edges": "자연 임지",
    # imya_change.json — changed
    "rows of dark blue solar photovoltaic panels on a slope cleared out of the forest": "태양광 패널",
    "a building with a metal or tiled roof and a cleared yard cut into the forested hillside": "건물+마당",
    "machine-graded bare earth with tire tracks and a new road or sharp cleared edge cut into the wooded slope": "기계 정지+절토",
    "a logged clearing of bare soil scattered with cut tree stumps among standing trees": "벌채 나지",
    "a plowed field, rows of orchard trees or white plastic greenhouse tunnels on former forest land": "개간 밭/과수",
    "rows of low rounded grassy grave mounds on a bare patch cleared in the forest": "분묘",
    "a quarry or construction excavation of bare earth, rock and gravel with concrete retaining walls on the slope": "채석/굴착",
    "an open dirt yard with parked vehicles, shipping containers or stacked materials surrounded by trees": "야적장",
    # nongji_change.json — farm
    "a bare plowed dry field with soil ready for planting": "갈아엎은 맨 밭",
    "a dry agricultural field with crop rows and furrows": "이랑·고랑 밭",
    "a flooded or wet rice paddy with low embankments": "물 댄 논",
    "a fallow field covered with dry weeds and stubble": "휴경지",
    "a harvested field of bare soil in winter": "수확후 겨울나지",
    "an orchard with rows of bare fruit trees on tilled ground": "과수원",
    "green leafy vegetable crops growing in a field": "잎채소밭",
    # nongji_change.json — built
    "a building or house with a roof standing in a field": "건물/주택",
    "a large warehouse or factory shed with a wide metal roof": "창고/공장",
    "a long livestock barn or animal shed with a metal roof": "축사",
    "rows of dark blue solar photovoltaic panels covering the field": "태양광 패널",
    "a paved or concrete-covered lot with parked vehicles": "포장부지+주차",
    "an open yard with stored shipping containers, pipes or stacked building materials": "야적장",
    "a raised pad of piled fill soil or earth dumped over the field": "성토 단",
    "machine-graded bare ground with tire tracks prepared for construction": "건설 정지 나지",
    "an asphalt road or concrete driveway across the farmland": "도로/진입로",
    # nongji_change.json — greenhouse
    "rows of white plastic greenhouse tunnels": "비닐하우스",
    "long arched plastic-covered greenhouse structures": "아치형 온실",
    "a dense cluster of vinyl greenhouses": "비닐하우스 군집",
}


def short_label(sentence: str, tag: str, i: int) -> str:
    g = SHORT.get(sentence) or (sentence[:16] + "…")
    return f"{tag}{i} {g}"


def load_eval(sheet, group):
    p = RUNS / f"{sheet}_{group}" / "4_eval.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def collect(group):
    out = []
    for s in SHEETS:
        e = load_eval(s, group)
        if not e or not e.get("ranking"):
            continue
        rk = e["ranking"]
        fsb = e.get("full_sweep_best") or {}
        act = e.get("at_config_tau") or {}
        out.append(dict(sheet=s, prev=rk["prevalence"], auroc=rk["auroc"], ap=rk["ap"],
                        f1_best=fsb.get("F1"), f1_fix=act.get("F1"),
                        tau_best=fsb.get("tau"), tau_fix=act.get("tau"),
                        rec20=rk["recall_at"]["20%"], lift20=rk["lift_at"]["20%"],
                        b80=rk["budget_to_recall"]["80%"], n=rk["n"], npos=rk["n_pos"]))
    return out


IMYA = collect("imya")
NONGJI = collect("nongji")


# ─────────────────────────────────────────────────────────────────────
# FIG 1 — 고정임계 F1 불안정 vs 랭킹(AUROC) 안정
# ─────────────────────────────────────────────────────────────────────
def fig1():
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5))
    fig.suptitle("CadaLens 검증 데이터(세종·경북·경기·충북·전북 7개 도엽) 성능 평가\n"
                 "RemoteCLIP ViT-B/32 zero-shot · 손라벨 필지 "
                 f"{sum(d['n'] for d in IMYA)+sum(d['n'] for d in NONGJI):,}개",
                 fontsize=13, fontweight="bold")

    for k, (data, title) in enumerate([(IMYA, "임야 (산지전용 스크리닝)"),
                                       (NONGJI, "농지 전·답·과 (농지전용 스크리닝)")]):
        a = ax[0][k]
        data_s = sorted(data, key=lambda d: d["prev"])
        x = np.arange(len(data_s))
        a.bar(x - 0.2, [d["f1_fix"] for d in data_s], 0.4, label="고정 임계값 τ", color=C_FIX)
        a.bar(x + 0.2, [d["f1_best"] for d in data_s], 0.4, label="도엽별 최적 τ (사후)", color=C_BEST)
        for i, d in enumerate(data_s):
            a.text(i, max(d["f1_fix"], d["f1_best"]) + 0.02, f"위반\n{d['prev']*100:.0f}%",
                   ha="center", va="bottom", fontsize=7.5, color="#555")
        a.set_xticks(x)
        a.set_xticklabels([SHEET_NAME[d["sheet"]] for d in data_s], fontsize=8)
        a.set_ylim(0, 1.0)
        a.set_ylabel("F1 (형질변경 클래스)")
        a.set_title(f"{title}\n고정 임계값 F1: 도엽마다 {min(d['f1_fix'] for d in data_s):.2f}~"
                    f"{max(d['f1_fix'] for d in data_s):.2f} — 위반비율 낮을수록 붕괴", fontsize=10)
        a.legend(fontsize=8, loc="upper left")
        a.grid(axis="y", alpha=0.3)

    a = ax[1][0]
    x = np.arange(len(SHEETS))
    im = {d["sheet"]: d for d in IMYA}
    no = {d["sheet"]: d for d in NONGJI}
    a.bar(x - 0.2, [im[s]["auroc"] if s in im else 0 for s in SHEETS], 0.4, label="임야 AUROC", color=C_NORM)
    a.bar(x + 0.2, [no[s]["auroc"] if s in no else 0 for s in SHEETS], 0.4, label="농지 AUROC", color=C_CHG)
    allau = [d["auroc"] for d in IMYA + NONGJI]
    a.axhline(np.mean(allau), ls="--", c="k", lw=1, label=f"평균 {np.mean(allau):.3f}")
    a.set_xticks(x); a.set_xticklabels([SHEET_NAME[s] for s in SHEETS], fontsize=8)
    a.set_ylim(0.5, 1.0); a.set_ylabel("AUROC (위험도 랭킹)")
    a.set_title(f"랭킹 성능(AUROC)은 도엽 무관 안정: {min(allau):.2f}~{max(allau):.2f}", fontsize=10)
    a.legend(fontsize=8, loc="lower right"); a.grid(axis="y", alpha=0.3)

    a = ax[1][1]
    alld = [(d, "임야", "o") for d in IMYA] + [(d, "농지", "^") for d in NONGJI]
    for d, g, m in alld:
        a.scatter(d["prev"], d["f1_fix"], marker=m, s=55, c=C_FIX, edgecolor="k", lw=0.5, zorder=3)
        a.scatter(d["prev"], d["auroc"], marker=m, s=55, c="#1565c0", edgecolor="k", lw=0.5, zorder=3)
    pv = np.array([d["prev"] for d, _, _ in alld])
    f1 = np.array([d["f1_fix"] for d, _, _ in alld])
    au = np.array([d["auroc"] for d, _, _ in alld])
    for y, c, lab in [(f1, C_FIX, "고정 τ F1"), (au, "#1565c0", "AUROC")]:
        z = np.polyfit(pv, y, 1)
        xs = np.linspace(pv.min(), pv.max(), 50)
        a.plot(xs, np.polyval(z, xs), c=c, lw=2, label=f"{lab} 추세")
    a.set_xlabel("도엽 내 실제 위반비율 (prevalence)")
    a.set_ylabel("성능")
    a.set_ylim(0, 1.0)
    a.set_title("위반비율이 낮은 도엽에서\n고정 τ F1은 무너지지만 AUROC는 유지", fontsize=10)
    a.legend(fontsize=8, loc="center right"); a.grid(alpha=0.3)
    a.text(0.02, 0.04, "○ 임야  △ 농지", transform=a.transAxes, fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fig1_f1_vs_ranking.png", dpi=140)
    plt.close(fig)
    print("fig1 done")


# ─────────────────────────────────────────────────────────────────────
# FIG 2 — recall@budget (CAP) 곡선
# ─────────────────────────────────────────────────────────────────────
def cap_curve(sheet, group):
    scp = RUNS / f"{sheet}_{group}" / "2_scores.csv"
    ljp = DATA / f"{sheet}_{group}" / "labels.json"
    if not scp.exists() or not ljp.exists():
        return None
    sc = list(csv.DictReader(open(scp, encoding="utf-8-sig")))
    lj = json.loads(ljp.read_text(encoding="utf-8"))
    lab = {r["pnu"]: r["label"] for r in lj["labels"]}
    col = "p_forest" if group == "imya" else "p_screen"
    rows = []
    for r in sc:
        g = lab.get(r["pnu"])
        if g in (None, "보류", "제외"):
            continue
        risk = (1 - float(r[col])) if group == "imya" else float(r[col])
        rows.append((risk, 1 if g == "형질변경" else 0))
    if not rows:
        return None
    rows.sort(key=lambda t: -t[0])
    y = np.array([t[1] for t in rows])
    n, npos = len(y), int(y.sum())
    cum = np.cumsum(y) / npos
    frac = (np.arange(n) + 1) / n
    return frac, cum, npos / n


def fig2():
    fig, ax = plt.subplots(2, 2, figsize=(13, 10), gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("우선순위 큐 관점 — “상위 N% 필지를 검토하면 실제 위반의 몇 %를 회수하나”\n"
                 "(CAP / recall-at-budget 곡선, 도엽별)", fontsize=13, fontweight="bold")
    for k, (group, title, cc) in enumerate([("imya", "임야", C_NORM), ("nongji", "농지 전·답·과", C_CHG)]):
        a = ax[0][k]
        for s in SHEETS:
            r = cap_curve(s, group)
            if r is None:
                continue
            frac, cum, prev = r
            a.plot(np.r_[0, frac], np.r_[0, cum], lw=1.6, alpha=0.9, label=SHEET_NAME[s].replace("\n", " "))
        a.plot([0, 1], [0, 1], "--", c="gray", lw=1, label="무작위")
        a.axvline(0.2, c="k", lw=0.8, ls=":")
        a.text(0.21, 0.03, "예산 20%", fontsize=8)
        a.set_xlim(0, 1); a.set_ylim(0, 1)
        a.set_xlabel("검토한 필지 비율 (예산)")
        a.set_ylabel("회수한 위반 비율 (recall)")
        a.set_title(title, fontsize=11)
        a.legend(fontsize=7.5, loc="lower right")
        a.grid(alpha=0.3)

        a2 = ax[1][k]
        data = IMYA if group == "imya" else NONGJI
        data_s = sorted(data, key=lambda d: d["prev"])
        x = np.arange(len(data_s))
        a2.bar(x - 0.2, [d["rec20"] for d in data_s], 0.4, color=cc, label="rec@20%")
        a2.bar(x + 0.2, [d["lift20"] / 5 for d in data_s], 0.4, color="#90a4ae", label="lift@20% (÷5)")
        for i, d in enumerate(data_s):
            a2.text(i - 0.2, d["rec20"] + 0.02, f"{d['rec20']*100:.0f}", ha="center", fontsize=7)
            a2.text(i + 0.2, d["lift20"] / 5 + 0.02, f"{d['lift20']:.1f}×", ha="center", fontsize=7)
        a2.set_xticks(x); a2.set_xticklabels([SHEET_NAME[d["sheet"]] for d in data_s], fontsize=7.5)
        a2.set_ylim(0, 1.05)
        a2.set_title(f"예산 20%에서 회수율 / 향상도(lift) — 평균 rec {np.mean([d['rec20'] for d in data_s])*100:.0f}%, "
                     f"lift {np.mean([d['lift20'] for d in data_s]):.1f}×", fontsize=9)
        a2.legend(fontsize=8); a2.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig2_recall_budget.png", dpi=140)
    plt.close(fig)
    print("fig2 done")


# ─────────────────────────────────────────────────────────────────────
# FIG 3 — 세종 전량 티어 큐 검증
# ─────────────────────────────────────────────────────────────────────
def fig3():
    p = RUNS / "sejong_summary.json"
    if not p.exists():
        print("fig3 skip — sejong_summary.json 없음")
        return
    summ = json.loads(p.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle("세종시 전량 스크리닝 — 심각도 티어 우선순위 큐 검증 (손라벨 필지)", fontsize=13, fontweight="bold")

    a = ax[0]
    groups = [s["group"] for s in summ]
    tiers = ["T1", "T2", "T3"]
    tcol = {"T1": "#c62828", "T2": "#ef6c00", "T3": "#90a4ae"}
    x = np.arange(len(groups))
    for j, t in enumerate(tiers):
        vals = [s["tier_precision"].get(t, {}).get("precision", 0) for s in summ]
        ns = [s["tier_precision"].get(t, {}).get("n", 0) for s in summ]
        a.bar(x + (j - 1) * 0.26, vals, 0.26, color=tcol[t], label=t)
        for i, (v, nn) in enumerate(zip(vals, ns)):
            if nn:
                a.text(x[i] + (j - 1) * 0.26, v + 0.02, f"{v:.2f}\n(n={nn})", ha="center", fontsize=7.5)
    a.set_xticks(x); a.set_xticklabels(["임야", "농지"]); a.set_ylim(0, 1.0)
    a.set_ylabel("티어 내 실제 형질변경 비율 (정밀도)")
    a.set_title("티어별 정밀도\nT1·T2 = 감독관 우선검토 리스트", fontsize=10)
    a.legend(fontsize=9); a.grid(axis="y", alpha=0.3)

    a = ax[1]
    for s, c in zip(summ, [C_NORM, C_CHG]):
        rb = s["recall_at_budget"]
        xs = [int(k[:-1]) for k in rb]
        a.plot(xs, [rb[k] for k in rb], "o-", c=c, lw=2, label=f"{'임야' if s['group']=='imya' else '농지'}")
    a.plot([0, 100], [0, 1], "--", c="gray", lw=1, label="무작위")
    a.set_xlabel("검토 예산 (%)"); a.set_ylabel("위반 회수율")
    a.set_title("티어 정렬 큐의 recall@budget", fontsize=10)
    a.legend(fontsize=9); a.grid(alpha=0.3)

    a = ax[2]
    a.axis("off")
    lines = ["세종 전량 결과 요약", ""]
    for s in summ:
        g = "임야" if s["group"] == "imya" else "농지"
        tp = s["tier_precision"]
        t12n = sum(tp.get(t, {}).get("n", 0) for t in ("T1", "T2"))
        t12v = sum(tp.get(t, {}).get("형질변경", 0) for t in ("T1", "T2"))
        lines.append(f"[{g}]  전체 {s['n']:,}필지 (라벨 {s['labeled']['n']}, 위반 {s['labeled']['prevalence']*100:.0f}%)")
        lines.append(f"   티어 분포  T1={s['tiers']['T1']}  T2={s['tiers']['T2']}  T3={s['tiers']['T3']:,}")
        if t12n:
            lines.append(f"   T1+T2 상위 {t12n}필지 중 {t12v} 실제위반  →  정밀도 {t12v/t12n*100:.0f}%")
        lines.append(f"   예산 20% → 위반 {s['recall_at_budget']['20%']*100:.0f}% 회수 / 50% → {s['recall_at_budget']['50%']*100:.0f}%")
        lines.append("")
    lines.append("한계: T3에 임야 위반 잔존 — T3도 위험도순으로 훑어야 함")
    a.text(0, 1, "\n".join(lines), va="top", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / "fig3_tier_queue.png", dpi=140)
    plt.close(fig)
    print("fig3 done")


# ─────────────────────────────────────────────────────────────────────
# FIG 4/5 — RemoteCLIP 프롬프트별 코사인 유사도 (예시 필지)
# ─────────────────────────────────────────────────────────────────────
def load_prompt(fn):
    d = json.loads((ROOT / "03_pipeline" / "prompts" / fn).read_text(encoding="utf-8"))
    return d["groups"], d["softmax"]


def parse_sims(s):
    return [float(x) for x in s.split()] if s else []


def clip_examples(run_dir, chip_dir, labels_json, prompt_fn, group, picks_spec, outname, title):
    sc = {r["pnu"]: r for r in csv.DictReader(open(RUNS / run_dir / "2_scores.csv", encoding="utf-8-sig"))}
    lj = json.loads((DATA / labels_json).read_text(encoding="utf-8"))
    lab = {r["pnu"]: r["label"] for r in lj["labels"]}
    groups, softmax = load_prompt(prompt_fn)
    g_norm, g_chg = softmax[0], softmax[1]
    norm_p, chg_p = groups[g_norm], groups[g_chg]
    gh_p = groups.get("greenhouse", [])

    col = "p_forest" if group == "imya" else "p_screen"
    cand = []
    for pnu, r in sc.items():
        g = lab.get(pnu)
        if g in (None, "보류", "제외"):
            continue
        chp = ROOT / "03_pipeline" / chip_dir / r["chip"]
        if not chp.exists():
            continue
        cand.append((pnu, r, g, float(r[col]), chp))
    chosen, used = [], set()
    for want_lab, kind in picks_spec:
        pool = [c for c in cand if c[2] == want_lab and c[0] not in used]
        if not pool:
            continue
        if kind == "high_pforest":
            pool.sort(key=lambda c: -c[3])
        elif kind == "low_pforest":
            pool.sort(key=lambda c: c[3])
        else:
            pool.sort(key=lambda c: abs(c[3] - 0.5))
        chosen.append(pool[0])
        used.add(pool[0][0])

    norm_tag = "숲" if group == "imya" else "farm"
    chg_tag = "변경" if group == "imya" else "built"

    nrow = len(chosen)
    nbar = len(norm_p) + len(chg_p) + (1 if (gh_p and any(c[1].get("gh_sim") for c in chosen)) else 0)
    row_h = max(3.2, 0.34 * nbar + 1.1)
    fig = plt.figure(figsize=(12, row_h * nrow))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for i, (pnu, r, g, pv, chp) in enumerate(chosen):
        axi = fig.add_subplot(nrow, 4, i * 4 + 1)
        try:
            import matplotlib.image as mpimg
            axi.imshow(mpimg.imread(chp))
        except Exception as e:
            axi.text(0.5, 0.5, str(e), fontsize=6)
        axi.set_xticks([]); axi.set_yticks([])
        pcol = C_CHG if g == "형질변경" else C_NORM
        axi.set_title(f"{r['jibun']}\n손라벨: {g}", color=pcol, fontsize=9)
        for sp in axi.spines.values():
            sp.set_edgecolor(pcol); sp.set_linewidth(2.5)

        axb = fig.add_subplot(nrow, 4, (i * 4 + 2, i * 4 + 4))
        sims_n = parse_sims(r.get("sims_normal", ""))
        sims_c = parse_sims(r.get("sims_changed", ""))
        labels_all, vals, cols = [], [], []
        for j, (t, v) in enumerate(zip(norm_p, sims_n), 1):
            labels_all.append(short_label(t, norm_tag, j)); vals.append(v); cols.append(C_NORM)
        for j, (t, v) in enumerate(zip(chg_p, sims_c), 1):
            labels_all.append(short_label(t, chg_tag, j)); vals.append(v); cols.append(C_CHG)
        if gh_p and r.get("gh_sim"):
            labels_all.append(short_label(gh_p[0], "GH", 1)); vals.append(float(r["gh_sim"])); cols.append(C_GH)
        yy = np.arange(len(labels_all))
        axb.barh(yy, vals, color=cols)
        axb.set_yticks(yy); axb.set_yticklabels(labels_all, fontsize=8.5)
        axb.invert_yaxis()
        axb.set_xlabel("이미지-텍스트 코사인 유사도", fontsize=8)
        axb.set_xlim(0, max(vals) * 1.16)
        mx_n = max(sims_n) if sims_n else 0
        mx_c = max(sims_c) if sims_c else 0
        pf = float(r["p_forest"]); ps = float(r["p_screen"])
        axb.axvline(mx_n, c=C_NORM, ls=":", lw=1)
        axb.axvline(mx_c, c=C_CHG, ls=":", lw=1)
        verdict = "형질변경의심" if ((group == "imya" and pf < 0.85) or (group != "imya" and ps >= 0.6)) else "정상"
        axb.set_title(f"정상 최고 {mx_n:.3f} · 변경 최고 {mx_c:.3f}  →  "
                      f"p_정상={pf:.2f} · p_변경={ps:.2f}   ⇒ {verdict}", fontsize=9)
        for j, v in enumerate(vals):
            axb.text(v + max(vals) * 0.01, j, f"{v:.3f}", va="center", fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / outname, dpi=140)
    plt.close(fig)
    print(outname, "done")


# ─────────────────────────────────────────────────────────────────────
# FIG 6 — 프롬프트 최고유사도 분리도
# ─────────────────────────────────────────────────────────────────────
def fig6():
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("RemoteCLIP: ‘정상 프롬프트 최고유사도’ vs ‘형질변경 프롬프트 최고유사도’ — 전 검증필지\n"
                 "두 최고값의 상대크기(softmax)가 p값을 만든다. 분포가 겹칠수록 고정 임계값이 불안정",
                 fontsize=12, fontweight="bold")
    for k, (group, gtitle) in enumerate([("imya", "임야"), ("nongji", "농지 전·답·과")]):
        xs_n, xs_c, cols = [], [], []
        pf_norm, pf_chg = [], []
        for s in SHEETS:
            rd = RUNS / f"{s}_{group}"
            if not (rd / "2_scores.csv").exists():
                continue
            lj = json.loads((DATA / f"{s}_{group}" / "labels.json").read_text(encoding="utf-8"))
            lab = {r["pnu"]: r["label"] for r in lj["labels"]}
            for r in csv.DictReader(open(rd / "2_scores.csv", encoding="utf-8-sig")):
                g = lab.get(r["pnu"])
                if g in (None, "보류", "제외"):
                    continue
                cn = parse_sims(r.get("sims_normal", "")); cc = parse_sims(r.get("sims_changed", ""))
                if not cn or not cc:
                    continue
                is_chg = g == "형질변경"
                xs_n.append(max(cn)); xs_c.append(max(cc))
                cols.append(C_CHG if is_chg else C_NORM)
                (pf_chg if is_chg else pf_norm).append(float(r["p_forest"]))
        a = ax[0][k]
        a.scatter(xs_n, xs_c, c=cols, s=8, alpha=0.5, edgecolor="none")
        lim = [min(xs_n + xs_c), max(xs_n + xs_c)]
        a.plot(lim, lim, "k--", lw=1)
        a.set_xlabel("정상(임야/farm) 프롬프트 최고 코사인")
        a.set_ylabel("형질변경(changed/built) 프롬프트 최고 코사인")
        a.set_title(f"{gtitle} — 대각선 위=변경 우세  (주황=실제 위반, 초록=정상)", fontsize=10)
        a.grid(alpha=0.3)

        a = ax[1][k]
        bins = np.linspace(0, 1, 41)
        a.hist(pf_norm, bins=bins, color=C_NORM, alpha=0.6, label=f"정상 (n={len(pf_norm)})", density=True)
        a.hist(pf_chg, bins=bins, color=C_CHG, alpha=0.6, label=f"형질변경 (n={len(pf_chg)})", density=True)
        thr = 0.85 if group == "imya" else 0.4
        a.axvline(thr, c="k", ls="--", lw=1.2, label="고정 임계값 근방")
        a.set_xlabel("p_정상 (= softmax 정상확률)")
        a.set_ylabel("밀도")
        a.set_title(f"{gtitle} — 클래스별 p_정상 분포", fontsize=10)
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig6_clip_separation.png", dpi=140)
    plt.close(fig)
    print("fig6 done")


# ─────────────────────────────────────────────────────────────────────
# FIG 7 — 우선순위 매트릭스 (신뢰도 × 심각도)
# ─────────────────────────────────────────────────────────────────────
def fig7_matrix(run_dir="sejong_36710022_nongji", chip_dir="data/36710022_nongji",
                risk_floor=0.5, sev_hi=58, outname="fig7_priority_matrix.png",
                title="조사 우선순위 = 신뢰도 × 심각도  (세종 36710022 · 농지 2,102필지)"):
    import matplotlib.image as mpimg
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    wl = list(csv.DictReader(open(RUNS / run_dir / "5_worklist.csv", encoding="utf-8-sig")))
    risk = np.array([float(r["risk"]) for r in wl])
    sev = np.array([float(r["severity_score"]) for r in wl])
    tier = np.array([r["tier"] for r in wl])
    tcol = {"T1": "#c62828", "T2": "#ef6c00", "T3": "#9e9e9e"}

    fig, ax = plt.subplots(figsize=(13.5, 7.8))
    fig.subplots_adjust(left=0.065, right=0.63, top=0.9, bottom=0.1)
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.36)

    # 4분면 배경 (데이터 좌표 span)
    ax.axvspan(0, risk_floor, color="#9e9e9e", alpha=0.10, zorder=0)
    ax.add_patch(plt.Rectangle((risk_floor, sev_hi), 1 - risk_floor, 100 - sev_hi,
                               color="#c62828", alpha=0.08, zorder=0))
    ax.add_patch(plt.Rectangle((risk_floor, 0), 1 - risk_floor, sev_hi,
                               color="#ef6c00", alpha=0.06, zorder=0))
    ax.axvline(risk_floor, color="#555", lw=1, ls="--", zorder=1)
    ax.plot([risk_floor, 1.0], [sev_hi, sev_hi], color="#555", lw=1, ls="--", zorder=1)

    # 대각 화살표 — 조사 순서 (산점 뒤)
    ax.annotate("", xy=(0.54, 6), xytext=(0.99, 92),
                arrowprops=dict(arrowstyle="-|>", lw=5, color="#1565c0", alpha=0.16), zorder=1)
    ax.text(0.90, 82, "조사 순서", rotation=42, fontsize=10.5, color="#1565c0",
            fontweight="bold", ha="center", va="center", alpha=0.85)

    # 필지 산점 (티어색)
    for t in ("T3", "T2", "T1"):
        m = tier == t
        ax.scatter(risk[m], sev[m], s=13, c=tcol[t], alpha=0.4 if t == "T3" else 0.65,
                   edgecolor="none", label=f"{t}  ({m.sum():,})", zorder=3)

    # 분면 라벨 (모서리)
    ax.text(0.515, 99, "★ T1 · 즉시 조사", ha="left", va="top", fontsize=12,
            fontweight="bold", color="#c62828", zorder=5)
    ax.text(0.515, 94.5, "개발제한구역 + 신뢰도 높음", ha="left", va="top", fontsize=9, color="#c62828", zorder=5)
    ax.text(0.515, sev_hi - 3, "T2 · 현장 확인", ha="left", va="top", fontsize=11, color="#ef6c00", zorder=5)
    ax.text(0.515, sev_hi - 7.5, "규모 크거나 변형 넓음", ha="left", va="top", fontsize=9, color="#ef6c00", zorder=5)
    ax.text(0.02, 99, "후순위 (T3) · 신뢰도 낮음", ha="left", va="top", fontsize=10, color="#616161", zorder=5)

    # 실제 필지 썸네일 — 오른쪽 여백에 세로로 배치 + 실제 점으로 연결선
    def pick(cond):
        idx = np.where(cond)[0]
        return int(idx[0]) if len(idx) else None

    gbcol = np.array([r["in_greenbelt"] in ("True", "1", "true") for r in wl])
    picks = [
        (pick((tier == "T1") & (sev >= 66) & (risk > 0.95)), 0.80),
        (pick((tier == "T2") & (risk > 0.9) & (sev < sev_hi - 6) & (sev > 22)), 0.49),
        (pick((tier == "T3") & (risk < 0.32) & (risk > 0.14) & (sev < 28) & ~gbcol), 0.18),
    ]
    for i, fy in picks:
        if i is None:
            continue
        r = wl[i]
        chp = ROOT / "03_pipeline" / chip_dir / r["chip"]
        if not chp.exists():
            continue
        xy = (float(r["risk"]), float(r["severity_score"]))
        ec = tcol[r["tier"]]
        try:
            im = OffsetImage(mpimg.imread(chp), zoom=0.44)
        except Exception:
            continue
        ab = AnnotationBbox(im, xy, xybox=(1.12, fy), frameon=True, pad=0.2,
                            bboxprops=dict(edgecolor=ec, lw=2.4),
                            arrowprops=dict(arrowstyle="-", color=ec, lw=1.3, alpha=0.8),
                            xycoords="data", boxcoords=("axes fraction", "axes fraction"),
                            box_alignment=(0.5, 0.5), zorder=6, annotation_clip=False)
        ax.add_artist(ab)
        gb = " · GB저촉" if r["in_greenbelt"] in ("True", "1", "true") else ""
        ax.annotate(f'{r["tier"]} · {r["jibun"]}\n신뢰도 {xy[0]:.2f} · 심각도 {xy[1]:.0f}{gb}',
                    xy=(1.12, fy), xycoords="axes fraction", xytext=(0, 38),
                    textcoords="offset points", fontsize=8, ha="center", va="bottom",
                    color=ec, zorder=6, annotation_clip=False, linespacing=1.4)

    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 106)
    ax.set_xticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax.set_xlabel("신뢰도 (risk)  ←  형질변경 프롬프트와의 코사인 유사도", fontsize=10)
    ax.set_ylabel("심각도 (severity)  ←  개발제한구역 40 + 형질변경 면적비율 30 + 전환 규모 30", fontsize=10)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92, title="필지 티어")
    ax.grid(alpha=0.18)
    fig.savefig(OUT / outname, dpi=140)
    plt.close(fig)
    print(outname, "done")


def figs_clip():
    clip_examples("sejong_36709030_imya", "data/36709030_imya", "36709030_imya/labels.json",
                  "imya_change.json", "imya",
                  [("임야", "high_pforest"), ("형질변경", "low_pforest"),
                   ("형질변경", "mid"), ("임야", "low_pforest")],
                  "fig4_clip_imya.png",
                  "RemoteCLIP 프롬프트별 유사도 — 임야 예시 필지 (프롬프트: imya_change.json, 세종 공주030)")
    clip_examples("36813045_nongji", "data/36813045_nongji", "36813045_nongji/labels.json",
                  "nongji_change.json", "nongji",
                  [("농지", "low_pforest"), ("형질변경", "high_pforest"),
                   ("형질변경", "mid"), ("농지", "high_pforest")],
                  "fig5_clip_nongji.png",
                  "RemoteCLIP 프롬프트별 유사도 — 농지 예시 필지 (프롬프트: nongji_change.json, 김천045)")


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    if only == "clip":
        figs_clip()
    elif only == "matrix":
        fig7_matrix()
    else:
        fig1(); fig2(); fig3(); figs_clip(); fig6(); fig7_matrix()
    print("ALL DONE →", OUT)
