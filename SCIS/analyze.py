"""Aggregate all results into results/summary.json, results/tables.md and figures/*.png.
Every number in the manuscript's Section 6 comes from this script.

    python analyze.py
"""
import glob, json, os
import numpy as np
from scipy.stats import wilcoxon, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
os.makedirs("figures", exist_ok=True)

R = [json.load(open(f)) for f in sorted(glob.glob("results/main/seed_*.json"))]
S = len(R)
LED = json.load(open("results/ledger.json"))
SC = [json.load(open(f)) for f in sorted(glob.glob("results/scaling/*.json"))]

LABEL = {"iot_only": "IoT-only", "ai_only": "AI-only (tabular)", "bc_only": "Blockchain-only",
         "separate": "Separate (tabular)", "proposed": "Integrated (tabular)",
         "nn_separate": "Separate (neural)", "nn_proposed": "Proposed (neural)",
         "nn2opt": "NN + 2-opt", "ratio2opt": "Demand-aware + 2-opt"}
METRICS = [("cost", "Total cost per day", "{:.2f}"), ("cpu", "Cost per 1000 units", "{:.3f}"),
           ("ds", "Demand satisfaction", "{:.3f}"), ("delay", "Mean delivery delay (h)", "{:.3f}"),
           ("re", "Routing efficiency", "{:.3f}"), ("stops", "Customers served per day", "{:.1f}"),
           ("latency", "Decision latency (ms)", "{:.3f}")]
LOWER_BETTER = {"cost", "cpu", "delay", "latency"}


def col(cfg, m, rows=R):
    return np.array([r["results"][cfg][m] for r in rows])


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def wx(a, b):
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(a, b).pvalue)


def fmt_p(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


summ = {"n_seeds": S}
md = []

# ---------------------------------------------------------------- Table 4
cfgs4 = ["iot_only", "ai_only", "bc_only", "proposed", "nn_proposed", "nn2opt", "ratio2opt"]
t4 = {}
md.append("## Table 4. Overall performance (mean ± SD over %d seeds, 200 held-out days each)\n" % S)
md.append("| Metric | " + " | ".join(LABEL[c] for c in cfgs4) + " |")
md.append("|---|" + "---|" * len(cfgs4))
for m, name, f in METRICS:
    row = []
    for c in cfgs4:
        v = col(c, m); t4.setdefault(c, {})[m] = (float(v.mean()), float(v.std(ddof=1)))
        row.append(f"{f.format(v.mean())} ± {f.format(v.std(ddof=1))}")
    md.append(f"| {name} | " + " | ".join(row) + " |")
summ["table4"] = t4

# ---------------------------------------------------------------- Table 5: proposed (neural) vs baselines
base5 = ["iot_only", "bc_only", "ai_only", "proposed", "nn2opt", "ratio2opt"]
m5 = ["cost", "cpu", "ds", "delay"]
raw, keys = [], []
for b in base5:
    for m in m5:
        raw.append(wx(col("nn_proposed", m), col(b, m))); keys.append((b, m))
adj = dict(zip(keys, holm(raw)))
t5 = {}
md.append("\n## Table 5. Proposed (neural) relative to each baseline (%%; positive = proposed better; Holm-adjusted Wilcoxon p, %d paired seeds)\n" % S)
md.append("| Baseline | " + " | ".join(f"{dict((a, b) for a, b, _ in METRICS)[m]}" for m in m5) + " |")
md.append("|---|" + "---|" * len(m5))
for b in base5:
    cells = []
    for m in m5:
        p, q = col("nn_proposed", m).mean(), col(b, m).mean()
        rel = 100 * (q - p) / q if m in LOWER_BETTER else 100 * (p - q) / q
        t5.setdefault(b, {})[m] = dict(rel=float(rel), p_holm=float(adj[(b, m)]))
        cells.append(f"{rel:+.1f} ({fmt_p(adj[(b, m)])})")
    md.append(f"| {LABEL[b]} | " + " | ".join(cells) + " |")
summ["table5"] = t5

# ---------------------------------------------------------------- Table 6: integrated vs separate
t6 = {}
md.append("\n## Table 6. Integrated vs separate reward (mean ± SD; Holm-adjusted Wilcoxon p)\n")
md.append("| Router | Model | Cost/day | Cost per 1000 units | Demand satisfaction | Delay (h) |")
md.append("|---|---|---|---|---|---|")
pairs = [("tabular", "separate", "proposed"), ("neural", "nn_separate", "nn_proposed")]
raw, keys = [], []
for rt, sep, integ in pairs:
    for m in ["cost", "cpu", "ds", "delay"]:
        raw.append(wx(col(integ, m), col(sep, m))); keys.append((rt, m))
adj6 = dict(zip(keys, holm(raw)))
for rt, sep, integ in pairs:
    for lab, c in (("Separate", sep), ("Integrated", integ)):
        cells = [f"{col(c, m).mean():.3f} ± {col(c, m).std(ddof=1):.3f}" for m in ["cost", "cpu", "ds", "delay"]]
        md.append(f"| {rt} | {lab} | " + " | ".join(cells) + " |")
    md.append(f"| {rt} | p (Holm) | " + " | ".join(fmt_p(adj6[(rt, m)]).replace("p", "", 1) for m in ["cost", "cpu", "ds", "delay"]) + " |")
    t6[rt] = {m: dict(sep=float(col(sep, m).mean()), integ=float(col(integ, m).mean()),
                      p_holm=float(adj6[(rt, m)])) for m in ["cost", "cpu", "ds", "delay"]}
summ["table6"] = t6

# ---------------------------------------------------------------- Table 7: blockchain timing (measured)
t7 = {}
md.append("\n## Table 7. Measured blockchain validation cost (wall-clock, mean ± SD over seeds)\n")
md.append("| Configuration | Validation (ms) | Consensus + hashing (ms) | Total ledger (ms) | Decision latency (ms) | Ledger share of latency | Chains intact | Rejected plans |")
md.append("|---|---|---|---|---|---|---|---|")
for c in ["bc_only", "separate", "proposed", "nn_separate", "nn_proposed"]:
    v, k = col(c, "t_val"), col(c, "t_cons")
    tot = v + k
    lat, ov = col(c, "latency"), col(c, "bc_overhead")
    intact = sum(r["results"][c]["chain_intact"] for r in R)
    rej = sum(r["results"][c]["rejected"] for r in R)
    t7[c] = dict(val=float(v.mean()), cons=float(k.mean()), total=float(tot.mean()), latency=float(lat.mean()),
                 overhead=float(ov.mean()), intact=int(intact), rejected=int(rej))
    md.append(f"| {LABEL[c]} | {v.mean():.3f} ± {v.std(ddof=1):.3f} | {k.mean():.3f} ± {k.std(ddof=1):.3f} | "
              f"{tot.mean():.3f} | {lat.mean():.3f} | {100 * ov.mean():.1f}% | {intact}/{S} | {rej} |")
summ["table7"] = t7

# ---------------------------------------------------------------- Table 8: ledger security
tam = LED["tamper"]
t8 = dict(edit_no_rehash=float(np.mean([t["edit_no_rehash"] for t in tam])),
          edit_rehash_block=float(np.mean([t["edit_rehash_block"] for t in tam])),
          edit_rehash_suffix_single_copy=float(np.mean([t["edit_rehash_suffix_single_copy"] for t in tam])),
          injection={})
md.append("\n## Table 8a. Tamper detection (%d seeds x 200 trials; single ledger copy)\n" % len(tam))
md.append("| Attack | Detection rate |\n|---|---|")
md.append(f"| Edit a block, no re-hash | {100 * t8['edit_no_rehash']:.1f}% |")
md.append(f"| Edit and re-hash one block | {100 * t8['edit_rehash_block']:.1f}% |")
md.append(f"| Edit and re-hash the whole suffix (one copy) | {100 * t8['edit_rehash_suffix_single_copy']:.1f}% |")
md.append("\n## Table 8b. Invalid-decision injection (10%% corrupted proposals, 500 days, %d seeds; 5 validators)\n" % len(tam))
md.append("| Faulty validators | Invalid plans accepted | Valid plans rejected | Final trust (faulty) | Final trust (honest) | Faulty excluded (share of days) | Honest excluded |")
md.append("|---|---|---|---|---|---|---|")
for f, rows in LED["injection"].items():
    g = lambda k: (float(np.mean([r[k] for r in rows])) if rows[0][k] is not None else None)
    d = {k: g(k) for k in ["invalid_accept_rate", "valid_reject_rate", "final_trust_faulty", "final_trust_honest",
                           "share_faulty_excluded", "share_honest_excluded", "n_invalid"]}
    t8["injection"][f] = d
    fm = lambda x, pct=False: "—" if x is None else (f"{100 * x:.1f}%" if pct else f"{x:.3f}")
    md.append(f"| {f} of 5 | {fm(d['invalid_accept_rate'], 1)} | {fm(d['valid_reject_rate'], 1)} | "
              f"{fm(d['final_trust_faulty'])} | {fm(d['final_trust_honest'])} | {fm(d['share_faulty_excluded'], 1)} | "
              f"{fm(d['share_honest_excluded'], 1)} |")
summ["table8"] = t8

# ---------------------------------------------------------------- Table 9: scaling
t9 = {}
md.append("\n## Table 9. Scaling (customers bootstrapped from the real pool; 5 seeds per size; mean)\n")
md.append("| Customers | Vehicles | Q-table entries | Neural parameters | Cost/1000 units: tabular | neural | demand-aware + 2-opt | Neural vs heuristic |")
md.append("|---|---|---|---|---|---|---|---|")
for n in sorted({s["n"] for s in SC}):
    rows = [s for s in SC if s["n"] == n]
    q = np.mean([s["q_table_size"] for s in rows])
    tb = np.array([s["results"]["tabular"]["cpu"] for s in rows])
    nn = np.array([s["results"]["neural"]["cpu"] for s in rows])
    ra = np.array([s["results"]["ratio2opt"]["cpu"] for s in rows])
    wins = int((nn < ra).sum())
    rel = 100 * (ra.mean() - nn.mean()) / ra.mean()
    t9[n] = dict(K=rows[0]["K"], q=float(q), params=rows[0]["nn_params"], tab=float(tb.mean()), nn=float(nn.mean()),
                 ratio=float(ra.mean()), rel=float(rel), wins=wins, n_seeds=len(rows))
    md.append(f"| {n} | {rows[0]['K']} | {q:.0f} | {rows[0]['nn_params']} | {tb.mean():.3f} | {nn.mean():.3f} | "
              f"{ra.mean():.3f} | {rel:+.1f}% ({wins}/{len(rows)} seeds) |")
summ["table9"] = t9
ns = sorted(t9)
summ["scaling_growth"] = dict(q_ratio=t9[ns[-1]]["q"] / t9[ns[0]]["q"], n_ratio=ns[-1] / ns[0])

# ---------------------------------------------------------------- sensitivity
SR = [r for r in R if r["sensitivity"]]
levels = sorted(SR[0]["sensitivity"], key=float)
sens = {}
for c in ["iot_only", "nn2opt", "ratio2opt", "proposed", "nn_proposed"]:
    sens[c] = {}
    for m in ["cpu", "ds"]:
        arr = np.array([[r["sensitivity"][l][c][m] for l in levels] for r in SR])
        x = np.repeat([float(l) for l in levels], len(SR)).reshape(len(levels), -1).T.ravel()
        rho, p = spearmanr(x, arr.ravel())
        sens[c][m] = dict(mean=arr.mean(0).tolist(), rho=float(rho), p=float(p))
gap = {}
for l in levels:
    a = np.array([r["sensitivity"][l]["nn_proposed"]["cpu"] for r in SR])
    b = np.array([r["sensitivity"][l]["ratio2opt"]["cpu"] for r in SR])
    gap[l] = dict(rel=float(100 * (b.mean() - a.mean()) / b.mean()), p=wx(a, b), wins=int((a < b).sum()))
summ["sensitivity"] = dict(levels=[float(l) for l in levels], by_config=sens, neural_vs_ratio=gap, n_seeds=len(SR))
md.append("\n## Sensitivity (%d seeds, 100 days per level): cost per 1000 units\n" % len(SR))
md.append("| Noise level | " + " | ".join(LABEL[c] for c in sens) + " | Neural vs demand-aware (p) |")
md.append("|---|" + "---|" * (len(sens) + 1))
for i, l in enumerate(levels):
    md.append(f"| {float(l):.0%} | " + " | ".join(f"{sens[c]['cpu']['mean'][i]:.3f}" for c in sens) +
              f" | {gap[l]['rel']:+.1f}% ({fmt_p(gap[l]['p'])}) |")

# ---------------------------------------------------------------- misc
summ["q_table_size"] = {k: float(np.mean([r["q_table_size"][k] for r in R])) for k in ("tab_joint", "tab_sep")}
summ["nn_params"] = R[0]["nn_params"]
summ["forecast_mae"] = float(np.mean([r["forecast_mae"] for r in R]))
summ["world"] = R[0]["world"]
curves = {k: np.array([r["curves"][k] for r in R]).mean(0).tolist() for k in R[0]["curves"]}
summ["curves"] = curves
json.dump(summ, open("results/summary.json", "w"), indent=1)
open("results/tables.md", "w").write("\n".join(md) + "\n")

# ================================================================ figures
C = {"iot_only": "#4C72B0", "ai_only": "#DD8452", "bc_only": "#55A868", "separate": "#C44E52",
     "proposed": "#8172B3", "nn_separate": "#DA8BC3", "nn_proposed": "#937860", "nn2opt": "#8C8C8C",
     "ratio2opt": "#CCB974"}
cf = ["iot_only", "ai_only", "bc_only", "proposed", "nn_proposed", "nn2opt", "ratio2opt"]
fig, ax = plt.subplots(2, 3, figsize=(11, 6))
for a, (m, name, _) in zip(ax.ravel(), [METRICS[i] for i in (0, 1, 2, 3, 4, 6)]):
    mu = [col(c, m).mean() for c in cf]; sd = [col(c, m).std(ddof=1) for c in cf]
    a.bar(range(len(cf)), mu, yerr=sd, color=[C[c] for c in cf], capsize=3)
    a.set_xticks(range(len(cf))); a.set_xticklabels([LABEL[c] for c in cf], rotation=35, ha="right", fontsize=7)
    a.set_title(name)
fig.suptitle(f"Performance across configurations (mean ± SD, {S} seeds, held-out days)")
fig.tight_layout(); fig.savefig("figures/Figure6.png", dpi=600); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.9))
w = 0.2
for i, m in enumerate(m5):
    ax.bar(np.arange(len(base5)) + (i - 1.5) * w, [t5[b][m]["rel"] for b in base5], w,
           label=dict((a, b) for a, b, _ in METRICS)[m])
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(base5))); ax.set_xticklabels([LABEL[b] for b in base5], rotation=20, ha="right")
ax.set_ylabel("Proposed (neural) better by (%)")
ax.legend(fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
fig.tight_layout(); fig.savefig("figures/Figure7.png", dpi=600); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
for a, m in zip(ax, ["cpu", "ds", "delay"]):
    x = np.arange(2)
    a.bar(x - 0.18, [col("separate", m).mean(), col("nn_separate", m).mean()], 0.36, label="Separate",
          yerr=[col("separate", m).std(ddof=1), col("nn_separate", m).std(ddof=1)], capsize=3, color="#C44E52")
    a.bar(x + 0.18, [col("proposed", m).mean(), col("nn_proposed", m).mean()], 0.36, label="Integrated",
          yerr=[col("proposed", m).std(ddof=1), col("nn_proposed", m).std(ddof=1)], capsize=3, color="#8172B3")
    a.set_xticks(x); a.set_xticklabels(["Tabular", "Neural"]); a.set_title(dict((q, n) for q, n, _ in METRICS)[m])
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig("figures/Figure8.png", dpi=600); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
xb = np.arange(1, 11) * 80
for k, lab, c in (("tab_joint", "Tabular (integrated)", "#8172B3"), ("nn_joint", "Neural (integrated)", "#937860")):
    cv = np.array(curves[k])
    ax[0].plot(xb, cv[:, 2], "o-", label=lab, color=c); ax[1].plot(xb, cv[:, 1], "o-", label=lab, color=c)
ref = t4["ratio2opt"]
ax[0].axhline(ref["cpu"][0], ls="--", color="#CCB974", label="Demand-aware + 2-opt (test)")
ax[1].axhline(ref["ds"][0], ls="--", color="#CCB974")
ax[0].set_title("Cost per 1000 units during training"); ax[1].set_title("Demand satisfaction during training")
for a in ax: a.set_xlabel("Training day")
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig("figures/Figure9.png", dpi=600); plt.close(fig)

# Figure 15 (measured ledger time) is drawn by rev_analyze.py from the Table 14 data (figures/Figure15.png).

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
lv = [100 * float(l) for l in levels]
for c in sens:
    ax[0].plot(lv, sens[c]["cpu"]["mean"], "o-", label=LABEL[c], color=C[c])
    ax[1].plot(lv, sens[c]["ds"]["mean"], "o-", label=LABEL[c], color=C[c])
ax[0].set_title("Cost per 1000 units"); ax[1].set_title("Demand satisfaction")
for a in ax: a.set_xlabel("IoT noise level (%)")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=7, ncol=len(l), loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig("figures/Figure12.png", dpi=600); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(ns, [t9[n]["q"] for n in ns], "o-", label="Tabular Q-table entries", color="#8172B3")
ax[0].plot(ns, [t9[n]["params"] for n in ns], "s-", label="Neural parameters", color="#937860")
ax[0].set_xscale("log", base=2); ax[0].set_yscale("log"); ax[0].set_xlabel("Customers"); ax[0].legend(fontsize=7)
ax[0].set_title("Representation size")
for k, lab, c in (("tab", "Tabular", "#8172B3"), ("nn", "Neural", "#937860"), ("ratio", "Demand-aware + 2-opt", "#CCB974")):
    ax[1].plot(ns, [t9[n][k] for n in ns], "o-", label=lab, color=c)
ax[1].set_xscale("log", base=2); ax[1].set_xlabel("Customers"); ax[1].set_title("Cost per 1000 units")
ax[1].legend(fontsize=7)
for a in ax:
    a.set_xticks(ns); a.set_xticklabels([str(n) for n in ns]); a.minorticks_off()
fig.tight_layout(); fig.savefig("figures/Figure13.png", dpi=600); plt.close(fig)
print(open("results/tables.md").read())
