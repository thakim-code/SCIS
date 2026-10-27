"""Pairs results/rev_robust with results/rev_variants; writes results/rev_robust_summary.json."""
import glob, json, os
import numpy as np
from scipy.stats import wilcoxon

def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = [0] * m; run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj

def cmp(a, b):  # a better (lower) than b
    a, b = np.array(a), np.array(b)
    rel = 100 * (b.mean() - a.mean()) / b.mean()
    p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
    return dict(rel=rel, wins=int((a < b).sum()), n=len(a), p=float(p))

out = {}
for grp, base in (("constraints", "ratio_tw"), ("sens", "ratio2opt")):
    keys = sorted({f.split("__")[1] for f in glob.glob(f"results/rev_robust/{grp}__*.json")})
    for k in keys:
        R = {m: [] for m in ("ro4", "hy", "alns", "heur", "nn", "ds_ro4", "ds_heur", "ds_alns", "ds_hy",
                             "lat_ro4", "lat_hy", "rej", "delay_ro4")}
        for s in range(10):
            f = f"results/rev_robust/{grp}__{k}__s{s}.json"
            if not os.path.exists(f): break
            r = json.load(open(f))["results"]; o = json.load(open(f"results/rev_variants/{grp}__{k}__s{s}.json"))["results"]
            assert abs(r["nn_check"]["cpu"] - o["nn"]["cpu"]) < 1e-9
            R["ro4"].append(r["nn_rollout4"]["cpu"]); R["hy"].append(r["alns_from_rollout4"]["cpu"])
            R["alns"].append(o["alns"]["cpu"]); R["heur"].append(o[base]["cpu"]); R["nn"].append(o["nn"]["cpu"])
            R["ds_ro4"].append(r["nn_rollout4"]["ds"]); R["ds_heur"].append(o[base]["ds"])
            R["ds_alns"].append(o["alns"]["ds"]); R["ds_hy"].append(r["alns_from_rollout4"]["ds"])
            R["lat_ro4"].append(r["nn_rollout4"]["latency"]); R["lat_hy"].append(r["alns_from_rollout4"]["latency"])
            R["rej"].append(r["nn_rollout4"].get("rejected",0) + r["alns_from_rollout4"].get("rejected",0))
            R["delay_ro4"].append(r["nn_rollout4"]["delay"])
        if len(R["ro4"]) < 10: continue
        out[f"{grp}:{k}"] = dict(
            ro4_vs_heur=cmp(R["ro4"], R["heur"]), hy_vs_heur=cmp(R["hy"], R["heur"]),
            hy_vs_alns=cmp(R["hy"], R["alns"]), ro4_vs_nn=cmp(R["ro4"], R["nn"]), ro4_vs_alns=cmp(R["ro4"], R["alns"]),
            mean={m: float(np.mean(v)) for m, v in R.items()})
for c in ("ro4_vs_heur", "hy_vs_heur", "hy_vs_alns"):
    ks = list(out); adj = holm([out[k][c]["p"] for k in ks])
    for k, a in zip(ks, adj): out[k][c]["p_holm"] = a
json.dump(out, open("results/rev_robust_summary.json", "w"), indent=1)
for k, v in out.items():
    print(f"{k:22s} ro4-heur {v['ro4_vs_heur']['rel']:5.1f}% {v['ro4_vs_heur']['wins']}/10 ph={v['ro4_vs_heur']['p_holm']:.3f} | "
          f"hy-heur {v['hy_vs_heur']['rel']:5.1f}% {v['hy_vs_heur']['wins']}/10 | hy-alns {v['hy_vs_alns']['rel']:4.1f}% "
          f"{v['hy_vs_alns']['wins']}/10 ph={v['hy_vs_alns']['p_holm']:.3f} | ro4-alns {v['ro4_vs_alns']['rel']:5.1f}% | "
          f"ds ro4 {v['mean']['ds_ro4']:.3f} heur {v['mean']['ds_heur']:.3f} alns {v['mean']['ds_alns']:.3f} hy {v['mean']['ds_hy']:.3f} rej {sum(1 for _ in [0]) and v['mean']['rej']:.1f}")
