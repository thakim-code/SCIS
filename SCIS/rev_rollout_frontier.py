"""Rollout engine on the quality-latency plane (same worlds/days/routers as the frontier group of rev_variants).
    python rev_rollout_frontier.py --data DATA.xlsx
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_common import *
from scs.data import load
from scs.revision import RolloutPolicy
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
DATA = load(a.data); out = {}
os.makedirs("results/rev_rollout_frontier", exist_ok=True)
for n in (41, 80, 160):
    for s in range(5):
        path = f"results/rev_rollout_frontier/n{n}_s{s}.json"
        if os.path.exists(path):
            continue
        if n == 41:
            w = World(DATA, np.random.default_rng(10_000 + s))
            days, _ = make_days(w, np.random.default_rng(20_000 + s), 100)
            nn = NeuralQ(True, np.random.default_rng(30_000 + 100 * s + 2), n_max=w.n)
            train_diag(nn, w, np.random.default_rng(40_000 + s), 800)
        else:
            w = World(DATA, np.random.default_rng(90_000 + 100 * n + s), n_customers=n)
            days, _ = make_days(w, np.random.default_rng(91_000 + 100 * n + s), 30)
            nn = NeuralQ(True, np.random.default_rng(93_000 + 100 * n + s), n_max=w.n)
            train_diag(nn, w, np.random.default_rng(94_000 + 100 * n + s), 400)
        days = days[:30]
        res = {f"rollout{k}": evaluate2(w, days, RolloutPolicy(nn, k), False) for k in (2, 4, 8)}
        save(path, dict(n=n, seed=s, results=res))
        print(n, s, {k: (round(v["cpu"], 4), round(v["latency"], 1)) for k, v in res.items()}, flush=True)
