"""Revision: variant experiments (10 seeds each; same worlds and the first 100 held-out days of the
main protocol). The learned router is retrained in every scenario, as an operator would on its own
network. Groups:
  constraints - hard due-time windows; hard windows + heterogeneous fleet (capacities 0.5/1.0/1.5)
  sens        - one-factor-at-a-time changes to the assumed operating parameters
  coverage    - coverage bonus in the reward / objective / heuristic score (cost-coverage frontier)
  hparams     - learned-router hyperparameters and reward weights
  frontier    - decision-quality vs decision-latency (ALNS time budgets vs learned router), 41/80/160 customers

    python rev_variants.py --data DATA.xlsx --group sens --workers 2
"""
import argparse, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_common import *
from scs.data import load
from scs.policies import REWARD

DATA = None
SEEDS = range(10)
TRAIN, TEST = 800, 100

CONSTRAINTS = {"tw": dict(hard_tw=True), "tw_het": dict(hard_tw=True, cap_mix=(0.5, 1.0, 1.5))}
SENS = {"base": {}, "cap0.40": dict(cap_tightness=0.40), "cap0.70": dict(cap_tightness=0.70),
        "cap1.00": dict(cap_tightness=1.00), "K2": dict(K=2), "K5": dict(K=5),
        "clustered": dict(layout="clustered"), "area200": dict(area_km=200.0),
        "ckm0.04": dict(c_km=0.04), "ckm0.08": dict(c_km=0.08), "due0.67": dict(due_scale=0.67),
        "demshift1.3": dict(_test_dem_mult=1.3)}
COV = [0.0, 0.1, 0.25, 0.5, 1.0]
RATIO_LAM = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
HP = {"lr3e-4": dict(lr=3e-4), "lr3e-3": dict(lr=3e-3), "gamma0.8": dict(gamma=0.8),
      "gamma0.99": dict(gamma=0.99), "hidden16": dict(hidden=16), "hidden64": dict(hidden=64),
      "decay0.3": dict(decay_frac=0.3), "decay0.9": dict(decay_frac=0.9),
      "adem1.5": dict(rw=dict(REWARD, a_dem=1.5)), "adem6": dict(rw=dict(REWARD, a_dem=6.0)),
      "adelay0.25": dict(rw=dict(REWARD, a_delay=0.25)), "adelay1.0": dict(rw=dict(REWARD, a_delay=1.0))}
BUDGETS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


def world_days(seed, params, n=None):
    params = dict(params)
    test_mult = params.pop("_test_dem_mult", None)
    if n is None:
        w = World(DATA, np.random.default_rng(10_000 + seed), params=params)
        days, _ = make_days(w, np.random.default_rng(20_000 + seed), TEST)
    else:
        w = World(DATA, np.random.default_rng(90_000 + 100 * n + seed), n_customers=n, params=params)
        days = None
    return w, days, test_mult


def train_nn(w, seed, n_train=TRAIN, **kw):
    nn = NeuralQ(True, np.random.default_rng(30_000 + 100 * seed + 2), n_max=w.n, **kw)
    train_diag(nn, w, np.random.default_rng(40_000 + seed), n_train)
    return nn


def job(args):
    group, key, seed, out_dir = args
    path = os.path.join(out_dir, f"{group}__{key}__s{seed}.json")
    if os.path.exists(path):
        return
    t0 = time.time()
    res = {}
    if group == "constraints":
        w, days, _ = world_days(seed, CONSTRAINTS[key])
        nn = train_nn(w, seed)
        for name, p in (("ratio_tw", RatioTW()), ("nn", nn), ("nn_2opt", Post2Opt(nn)),
                        ("alns", ALNSPolicy(1e9, seed=95_000 + seed, max_iters=100))):
            res[name] = evaluate2(w, days, p, True)
    elif group == "sens":
        w, days, mult = world_days(seed, SENS[key])
        nn = train_nn(w, seed)
        if mult:
            w.p["dem_mult"] = mult
            days, _ = make_days(w, np.random.default_rng(20_000 + seed), TEST)
        for name, p in (("ratio2opt", Ratio2Opt()), ("nn2opt", NN2Opt()), ("nn", nn),
                        ("alns", ALNSPolicy(1e9, seed=95_000 + seed, max_iters=50))):
            res[name] = evaluate2(w, days, p, False)
    elif group == "coverage":
        w, days, _ = world_days(seed, {})
        days = days[:60]                       # first 60 held-out days for every method in this group
        if key.startswith("lam"):
            lam = float(key[3:])
            res["ratio"] = evaluate2(w, days, RatioCov(lam), False)
        else:
            ac = float(key[3:])
            nn = train_nn(w, seed, rw=dict(REWARD, a_cov=ac))
            res["nn"] = evaluate2(w, days, nn, False)
            res["alns"] = evaluate2(w, days, ALNSPolicy(1e9, seed=95_000 + seed, max_iters=50,
                                                        a_cov=ac, rw=dict(REWARD, a_cov=ac)), False)
    elif group == "hparams":
        w, days, _ = world_days(seed, {})
        nn = train_nn(w, seed, **HP[key])
        res["nn"] = evaluate2(w, days, nn, False)
    elif group == "frontier":
        n = int(key)
        if n == 41:
            w, days, _ = world_days(seed, {})
            nn = train_nn(w, seed)
        else:
            w, _, _ = world_days(seed, {}, n=n)
            days, _ = make_days(w, np.random.default_rng(91_000 + 100 * n + seed), 30)
            nn = NeuralQ(True, np.random.default_rng(93_000 + 100 * n + seed), n_max=w.n)
            train_diag(nn, w, np.random.default_rng(94_000 + 100 * n + seed), 400)
        days = days[:30]
        res["ratio2opt"] = evaluate2(w, days, Ratio2Opt(), False)
        res["nn"] = evaluate2(w, days, nn, False)
        res["nn_2opt"] = evaluate2(w, days, Post2Opt(nn), False)
        for b in BUDGETS:
            p = ALNSPolicy(b, seed=95_000 + seed)
            res[f"alns_{b}"] = evaluate2(w, days, p, False)
        res["n"] = n
    save(path, dict(group=group, key=key, seed=seed, results=res, runtime_s=time.time() - t0))
    print(f"{group} {key} s{seed} {time.time() - t0:.0f}s", flush=True)


def init(path):
    global DATA
    DATA = load(path)


def jobs_for(group, out):
    if group == "constraints":
        keys = list(CONSTRAINTS)
    elif group == "sens":
        keys = list(SENS)
    elif group == "coverage":
        keys = [f"cov{a}" for a in COV] + [f"lam{l}" for l in RATIO_LAM]
    elif group == "hparams":
        keys = list(HP)
    elif group == "frontier":
        return [(group, str(n), s, out) for n in (41, 80, 160) for s in range(5)]
    return [(group, k, s, out) for k in keys for s in SEEDS]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--out", default="results/rev_variants")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for g in a.group.split(","):
        jobs += jobs_for(g, a.out)
    with Pool(a.workers, initializer=init, initargs=(a.data,)) as P:
        list(P.imap_unordered(job, jobs))
