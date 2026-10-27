"""Reference points for the coverage frontier on the same 60 held-out days: nearest-neighbour dispatch
(IoT-only) and NN + 2-opt.   python rev_cov_refs.py --data DATA.xlsx"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_common import *
from scs.data import load
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
data = load(a.data); out = {}
for s in range(10):
    w = World(data, np.random.default_rng(10_000 + s))
    days, _ = make_days(w, np.random.default_rng(20_000 + s), 100)
    days = days[:60]
    out[s] = {"greedy": evaluate2(w, days, Greedy(), False), "nn2opt": evaluate2(w, days, NN2Opt(), False)}
json.dump(out, open("results/rev_cov_refs.json", "w"))
for m in ("greedy", "nn2opt"):
    print(m, {k: round(float(np.mean([out[s][m][k] for s in out])), 3) for k in ("cpu", "stops", "distinct_share", "jain", "ds")})
