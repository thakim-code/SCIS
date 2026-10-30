"""Statistics, tables and figures for the first-round revision.
    python rev_analyze.py            -> results/rev_summary.json, results/rev_tables.md, figures/Figure10-11, 14-15, S1
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = {}
MD = []
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})
C = dict(nn="#1f5fa6", alns="#c0392b", ratio="#2e8b57", tab="#8e6bbf", ort="#d68910", nn2="#7f8c8d",
         greedy="#95a5a6")


def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def wp(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if np.allclose(a, b):
        return 1.0
    return float(wilcoxon(a, b).pvalue)


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def ms(x, d=3):
    x = np.asarray(x, float)
    return f"{np.nanmean(x):.{d}f} ± {np.nanstd(x, ddof=1):.{d}f}"


# ------------------------------------------------------------------ main
mains = [json.load(open(f)) for f in sorted(glob.glob("results/rev_main/seed_*.json"))]
for m in mains:                                   # merge learned-engine improvements (rev_rollout.py)
    f = f"results/rev_rollout/seed_{m['seed']:03d}.json"
    if os.path.exists(f):
        r = json.load(open(f))["results"]
        for k in ("nn_rollout4", "nn_rollout8", "alns_from_rollout4"):
            m["results"][k] = r[k]
S = len(mains)
if S:
    R = lambda cfg, k: np.array([m["results"][cfg][k] for m in mains if cfg in m["results"]])
    cfgs = list(mains[0]["results"].keys())
    OUT["main"] = {c: {k: [float(np.nanmean(R(c, k))), float(np.nanstd(R(c, k), ddof=1)) if len(R(c, k)) > 1 else 0.0]
                       for k in ("cost", "cpu", "ds", "delay", "re", "stops", "latency", "latency_p95", "km_pu",
                                 "distinct_share", "jain", "max_gap_median", "never_served", "t_val", "t_cons")}
                   for c in cfgs}
    for c in cfgs:
        OUT["main"][c]["n_seeds"] = int(len(R(c, "cpu")))
    # published-method comparison (proposed neural vs each; paired; Holm over all tests)
    base = "nn_proposed"
    comp = ["nn2opt", "ratio2opt", "ortools", "alns", "nn_proposed_2opt"]
    tests, keys = [], []
    for b in comp:
        idx = [i for i, m in enumerate(mains) if b in m["results"]]
        for k in ("cpu", "ds", "delay", "km_pu"):
            a = np.array([mains[i]["results"][base][k] for i in idx]); bb = np.array([mains[i]["results"][b][k] for i in idx])
            tests.append(wp(a, bb)); keys.append((b, k, a, bb))
    adj = holm(tests)
    OUT["vs_published"] = {}
    for (b, k, a, bb), p in zip(keys, adj):
        rel = 100 * (bb.mean() - a.mean()) / bb.mean() if k != "ds" else 100 * (a.mean() - bb.mean()) / bb.mean()
        OUT["vs_published"].setdefault(b, {})[k] = dict(rel=float(rel), p=float(p), wins=int((a < bb).sum() if k != "ds" else (a > bb).sum()), n=len(a))
    # every engine vs OR-Tools (cost per 1000 units; paired over the seeds that ran OR-Tools)
    OUT["vs_ortools"] = {}
    idx = [i for i, m in enumerate(mains) if "ortools" in m["results"]]
    o = np.array([mains[i]["results"]["ortools"]["cpu"] for i in idx])
    for b in ("nn2opt", "ratio2opt", "nn_proposed", "nn_proposed_2opt", "nn_rollout4", "nn_rollout8", "alns",
              "alns_from_rollout4"):
        if all(b in mains[i]["results"] for i in idx):
            a = np.array([mains[i]["results"][b]["cpu"] for i in idx])
            OUT["vs_ortools"][b] = dict(rel=float(100 * (o.mean() - a.mean()) / o.mean()), p=wp(a, o),
                                        wins=int((a < o).sum()), n=len(idx))
    # ALNS vs others
    OUT["alns_vs"] = {}
    for b in ("ratio2opt", "nn_proposed", "nn_proposed_2opt"):
        a, bb = R("alns", "cpu"), R(b, "cpu")
        OUT["alns_vs"][b] = dict(rel=float(100 * (bb.mean() - a.mean()) / bb.mean()), p=wp(a, bb), wins=int((a < bb).sum()))
    # faults
    F = {}
    for key in mains[0]["faults"]:
        F[key] = {k: float(np.mean([m["faults"][key][k] for m in mains])) for k in ("cpu", "exec_invalid_share", "ds")}
        F[key]["rejected"] = float(np.mean([m["faults"][key].get("rejected", 0) for m in mains]))
    OUT["faults"] = F
    # diagnostics
    D = {}
    for name in ("tab_joint", "nn_joint"):
        ck = np.array([[c["cpu"] for c in m["diag"][name]["ckpt"]] for m in mains])
        D[name] = dict(ckpt_mean=ck.mean(0).tolist(), ckpt_sd=ck.std(0, ddof=1).tolist() if S > 1 else [0] * ck.shape[1],
                       epochs=[c["epoch"] for c in mains[0]["diag"][name]["ckpt"]],
                       eps=np.mean([m["diag"][name]["eps"] for m in mains], 0).tolist())
        if name == "nn_joint":
            D[name]["td"] = np.mean([m["diag"][name]["td_loss"] for m in mains], 0).tolist()
            D[name]["q"] = np.mean([m["diag"][name]["mean_q"] for m in mains], 0).tolist()
            D[name]["n_updates"] = float(np.mean([m["diag"][name]["n_updates"] for m in mains]))
            last3 = ck[:, -3:].mean(1); first = ck[:, 4]
            D[name]["ckpt_last3_vs_400_rel"] = float(100 * (first.mean() - last3.mean()) / first.mean())
            D[name]["ckpt_cv_last3"] = float(np.mean(ck[:, -3:].std(1) / ck[:, -3:].mean(1)))
    OUT["diag"] = D

    # Figure: learning diagnostics
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for name, col, lab in (("tab_joint", C["tab"], "Tabular Q"), ("nn_joint", C["nn"], "Neural (DQN)")):
        e = np.array(D[name]["epochs"]); mu = np.array(D[name]["ckpt_mean"]); sd = np.array(D[name]["ckpt_sd"])
        ax[0].plot(e, mu, color=col, label=lab, lw=1.5); ax[0].fill_between(e, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
    for cfg, col, lab in (("ratio2opt", C["ratio"], "Demand-aware + 2-opt (test)"), ("alns", C["alns"], "ALNS (test)")):
        ax[0].axhline(OUT["main"][cfg]["cpu"][0], color=col, ls="--", lw=1, label=lab)
    ax[0].set_xlabel("Training day"); ax[0].set_ylabel("Cost per 1000 units\n(greedy, validation days)")
    td = np.array(D["nn_joint"]["td"]); x = np.linspace(0, 1, len(td) + 1)[1:] * D["nn_joint"]["n_updates"] / 1000
    ax[1].plot(x, td, color=C["nn"], lw=1.5); ax[1].set_xlabel("Gradient updates (thousands)"); ax[1].set_ylabel("Huber TD loss")
    a2 = ax[1].twinx(); a2.plot(x, D["nn_joint"]["q"], color=C["alns"], lw=1, ls=":"); a2.set_ylabel("Mean Q", color=C["alns"])
    a2.spines["top"].set_visible(False)
    e = (np.arange(10) + 0.5) * 80
    ax[2].plot(e, D["nn_joint"]["eps"], color=C["nn"], marker="o", ms=3, lw=1.2, label="Neural")
    ax[2].plot(e, D["tab_joint"]["eps"], color=C["tab"], marker="o", ms=3, lw=1.2, label="Tabular")
    ax[2].set_xlabel("Training day"); ax[2].set_ylabel("Exploration rate ε"); ax[2].legend(frameon=False, fontsize=6)
    for i, a in enumerate(ax):
        a.set_title("(" + "abc"[i] + ")", loc="left", fontsize=8)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig("figures/Figure10.png", dpi=600); plt.close(fig)

# ------------------------------------------------------------------ variants
V = {}
for f in glob.glob("results/rev_variants/*.json"):
    d = json.load(open(f)); V.setdefault((d["group"], d["key"]), []).append(d)


def vget(group, key, method, k):
    rows = sorted(V.get((group, key), []), key=lambda d: d["seed"])
    return np.array([d["results"][method][k] for d in rows if method in d["results"]])


if any(g == "constraints" for g, _ in V):
    OUT["constraints"] = {}
    for key in ("tw", "tw_het"):
        if (("constraints", key)) not in V:
            continue
        OUT["constraints"][key] = {m: {k: float(np.nanmean(vget("constraints", key, m, k))) for k in
                                       ("cpu", "ds", "delay", "stops", "latency", "rejected", "km_pu")}
                                   for m in ("ratio_tw", "nn", "nn_2opt", "alns")}
        a, b = vget("constraints", key, "nn", "cpu"), vget("constraints", key, "ratio_tw", "cpu")
        c = vget("constraints", key, "alns", "cpu")
        OUT["constraints"][key]["nn_vs_ratio"] = dict(rel=float(100 * (b.mean() - a.mean()) / b.mean()), p=wp(a, b), wins=int((a < b).sum()), n=len(a))
        OUT["constraints"][key]["alns_vs_nn"] = dict(rel=float(100 * (a.mean() - c.mean()) / a.mean()), p=wp(a, c), wins=int((c < a).sum()), n=len(a))

if any(g == "sens" for g, _ in V):
    OUT["sens"] = {}
    for (g, key) in sorted(V):
        if g != "sens":
            continue
        r = {m: {k: float(np.nanmean(vget(g, key, m, k))) for k in ("cpu", "ds", "stops", "delay", "km_pu")}
             for m in ("ratio2opt", "nn2opt", "nn", "alns")}
        a, b, c = vget(g, key, "nn", "cpu"), vget(g, key, "ratio2opt", "cpu"), vget(g, key, "alns", "cpu")
        r["nn_vs_ratio"] = dict(rel=float(100 * (b.mean() - a.mean()) / b.mean()), wins=int((a < b).sum()), n=len(a), p=wp(a, b))
        r["alns_vs_ratio"] = dict(rel=float(100 * (b.mean() - c.mean()) / b.mean()), wins=int((c < b).sum()), n=len(a), p=wp(c, b))
        r["alns_vs_nn"] = dict(rel=float(100 * (a.mean() - c.mean()) / a.mean()), wins=int((c < a).sum()), n=len(a))
        OUT["sens"][key] = r
    ps = holm([OUT["sens"][k]["nn_vs_ratio"]["p"] for k in OUT["sens"]])
    for k, p in zip(OUT["sens"], ps):
        OUT["sens"][k]["nn_vs_ratio"]["p_holm"] = float(p)

if any(g == "coverage" for g, _ in V):
    cov = {"nn": [], "alns": [], "ratio": []}
    for (g, key), rows in V.items():
        if g != "coverage":
            continue
        for m in rows[0]["results"]:
            par = float(key[3:])
            cov[m].append(dict(par=par, **{k: float(np.nanmean(vget(g, key, m, k))) for k in
                                           ("cpu", "ds", "stops", "distinct_share", "jain", "max_gap_median", "km_pu", "cost")}))
    for m in cov:
        cov[m].sort(key=lambda x: x["par"])
    OUT["coverage"] = cov
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for m, col, lab in (("ratio", C["ratio"], "Demand-aware greedy + 2-opt (λ)"), ("nn", C["nn"], "Neural router (α₄)"),
                        ("alns", C["alns"], "ALNS (α₄)")):
        xs = [c["stops"] for c in cov[m]]; ys = [c["cpu"] for c in cov[m]]
        ax[0].plot(xs, ys, marker="o", ms=3.5, color=col, lw=1.3, label=lab)
        xs = [100 * c["distinct_share"] for c in cov[m]]
        ax[1].plot([c["jain"] for c in cov[m]], ys, marker="o", ms=3.5, color=col, lw=1.3, label=lab)
    if os.path.exists("results/rev_cov_refs.json"):
        RF = json.load(open("results/rev_cov_refs.json"))
        g = {k: float(np.mean([RF[s]["greedy"][k] for s in RF])) for k in ("cpu", "stops", "jain", "distinct_share", "ds", "km_pu")}
        OUT["coverage_ref_dispatch"] = g
        ax[0].scatter([g["stops"]], [g["cpu"]], marker="s", color=C["greedy"], label="Nearest-neighbour dispatch", zorder=3)
        ax[1].scatter([g["jain"]], [g["cpu"]], marker="s", color=C["greedy"], zorder=3)
    ax[0].set_xlabel("Customers served per day"); ax[0].set_ylabel("Cost per 1000 units")
    ax[1].set_xlabel("Service-frequency fairness (Jain index, 60 days)"); ax[1].set_ylabel("Cost per 1000 units")
    ax[0].legend(frameon=False, fontsize=6)
    for i, a in enumerate(ax):
        a.set_title("(" + "ab"[i] + ")", loc="left", fontsize=8)
    fig.tight_layout(); fig.savefig("figures/Figure11.png", dpi=600); plt.close(fig)

if any(g == "hparams" for g, _ in V):
    OUT["hparams"] = {}
    base_nn = vget("sens", "base", "nn", "cpu") if ("sens", "base") in V else None
    base_ratio = vget("sens", "base", "ratio2opt", "cpu") if ("sens", "base") in V else None
    for (g, key) in sorted(V):
        if g != "hparams":
            continue
        a = vget(g, key, "nn", "cpu")
        seeds = sorted(d["seed"] for d in V[(g, key)])
        r = dict(cpu=float(a.mean()), sd=float(a.std(ddof=1)), ds=float(vget(g, key, "nn", "ds").mean()), n=len(a))
        if base_ratio is not None:
            br, bn = base_ratio[seeds], base_nn[seeds]
            r["vs_ratio"] = float(100 * (br.mean() - a.mean()) / br.mean())
            r["wins_vs_ratio"] = int((a < br).sum())
            r["vs_default"] = float(100 * (a.mean() - bn.mean()) / bn.mean())
        OUT["hparams"][key] = r

if any(g == "frontier" for g, _ in V):
    OUT["frontier"] = {}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for i, n in enumerate(("41", "80", "160")):
        if ("frontier", n) not in V:
            continue
        rows = V[("frontier", n)]
        get = lambda m, k: float(np.mean([d["results"][m][k] for d in rows]))
        bud = [k for k in rows[0]["results"] if k.startswith("alns_")]
        fr = dict(alns=[dict(budget=float(k[5:]), lat=get(k, "latency"), cpu=get(k, "cpu")) for k in bud],
                  nn=dict(lat=get("nn", "latency"), cpu=get("nn", "cpu")),
                  nn_2opt=dict(lat=get("nn_2opt", "latency"), cpu=get("nn_2opt", "cpu")),
                  ratio2opt=dict(lat=get("ratio2opt", "latency"), cpu=get("ratio2opt", "cpu")), seeds=len(rows))
        OUT["frontier"][n] = fr
        a = ax[i]
        a.plot([x["lat"] for x in fr["alns"]], [x["cpu"] for x in fr["alns"]], marker="o", ms=3, color=C["alns"], lw=1.3, label="ALNS (time budget)")
        a.scatter([fr["nn"]["lat"]], [fr["nn"]["cpu"]], color=C["nn"], zorder=3, label="Neural router")
        a.scatter([fr["nn_2opt"]["lat"]], [fr["nn_2opt"]["cpu"]], color=C["nn"], marker="^", zorder=3, label="Neural + 2-opt")
        a.scatter([fr["ratio2opt"]["lat"]], [fr["ratio2opt"]["cpu"]], color=C["ratio"], marker="s", zorder=3, label="Demand-aware + 2-opt")
        RF = [json.load(open(f)) for f in sorted(glob.glob(f"results/rev_rollout_frontier/n{n}_s*.json"))]
        if RF:
            fr["rollout"] = [dict(k=int(k[7:]), lat=float(np.mean([r["results"][k]["latency"] for r in RF])),
                                  cpu=float(np.mean([r["results"][k]["cpu"] for r in RF]))) for k in ("rollout2", "rollout4", "rollout8")]
            a.plot([x["lat"] for x in fr["rollout"]], [x["cpu"] for x in fr["rollout"]], marker="D", ms=3.5, color=C["nn"], lw=1.3,
                   ls="--", label="Neural + rollout (K = 2, 4, 8)")
        a.set_xscale("log"); a.set_xlabel("Decision latency per day (ms)")
        a.set_title(f"({'abc'[i]}) {n} customers", loc="left", fontsize=8)
        if i == 0:
            a.set_ylabel("Cost per 1000 units"); a.legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig("figures/Figure14.png", dpi=600); plt.close(fig)

# ------------------------------------------------------------------ ledger
if os.path.exists("results/rev_ledger.json"):
    L = json.load(open("results/rev_ledger.json"))
    OUT["bft"] = L["bft_scaling"]
    T = {}
    for k, v in L["trust"].items():
        T[k] = {kk: (float(np.mean([x[kk] for x in v])) if v[0][kk] is not None else None) for kk in v[0]}
    OUT["trust"] = T
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    runs = L["bft_scaling"]["41"]["runs"]
    Ns = [4, 7, 10, 16, 31, 64, 100]
    for prof, col in (("LAN", C["nn"]), ("WAN-regional", C["ratio"]), ("WAN-global", C["alns"])):
        ax[0].plot(Ns, [runs[f"{prof}|{N}|0"]["lat_med"] for N in Ns], marker="o", ms=3, color=col, label=prof)
        ax[0].plot(Ns, [runs[f"{prof}|{N}|{(N - 1) // 3}"]["lat_med"] for N in Ns], marker="x", ms=3, color=col, ls="--")
    ax[0].plot([], [], color="0.4", ls="--", marker="x", ms=3, label="f validators silent")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("Validators N"); ax[0].set_ylabel("Commit latency (ms, median)")
    ax[0].legend(frameon=False, fontsize=6); ax[0].set_xticks(Ns); ax[0].set_xticklabels(Ns)
    ax[1].plot(Ns, [runs[f"LAN|{N}|0"]["msgs"] for N in Ns], marker="o", ms=3, color=C["nn"], label="Messages per decision")
    a2 = ax[1].twinx(); a2.plot(Ns, [runs[f"LAN|{N}|0"]["cpu_max_ms"] for N in Ns], marker="s", ms=3, color=C["alns"], label="CPU per node (ms)")
    a2.set_ylabel("CPU per node per decision (ms)", color=C["alns"]); a2.spines["top"].set_visible(False)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("Validators N"); ax[1].set_ylabel("Messages per decision")
    ax[1].set_xticks(Ns); ax[1].set_xticklabels(Ns)
    for i, a in enumerate(ax):
        a.set_title("(" + "ab"[i] + ")", loc="left", fontsize=8)
    fig.tight_layout(); fig.savefig("figures/FigureS1.png", dpi=600); plt.close(fig)

json.dump(OUT, open("results/rev_summary.json", "w"), indent=1)
print(json.dumps({k: (v if k in ("vs_published", "alns_vs", "constraints") else "...") for k, v in OUT.items()}, indent=1)[:6000])

# ------------------------------------------------------------------ Figure 15: measured ledger time (Table 14 values)
if S:
    LBL = {"bc_only": "Blockchain-only", "separate": "Separate (tabular)", "proposed": "Integrated (tabular)",
           "nn_separate": "Separate (neural)", "nn_proposed": "Proposed (neural)"}
    cb = list(LBL)
    with plt.rc_context({"font.family": "serif", "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": True, "axes.spines.right": True}):
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        val = [OUT["main"][c]["t_val"][0] for c in cb]; cons = [OUT["main"][c]["t_cons"][0] for c in cb]
        ax.bar(range(len(cb)), val, label="Validation (5 validators)", color="#4C72B0")
        ax.bar(range(len(cb)), cons, bottom=val, label="Consensus + hashing", color="#55A868")
        ax.set_xticks(range(len(cb))); ax.set_xticklabels([LBL[c] for c in cb], rotation=20, ha="right")
        ax.set_ylabel("Measured time per daily decision (ms)")
        ax.legend(fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
        fig.tight_layout(); fig.savefig("figures/Figure15.png", dpi=600); plt.close(fig)
