"""Revision: robustness of the proposed engines (learned router + decision-time rollout, K = 4, and ALNS
seeded by it) under the same scenarios as rev_variants.py (sens, constraints). Same worlds, held-out days
and trained router (identical RNG streams), so results pair with results/rev_variants.
    python rev_robust.py --data DATA.xlsx --workers 2
"""
import argparse, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_common import *
from scs.data import load
from scs.revision import RolloutPolicy
import rev_variants as RV


def job(args):
    group, key, seed, out = args
    path = os.path.join(out, f"{group}__{key}__s{seed}.json")
    if os.path.exists(path):
        return
    t0 = time.time()
    params = RV.CONSTRAINTS[key] if group == "constraints" else RV.SENS[key]
    w, days, mult = RV.world_days(seed, params)
    nn = RV.train_nn(w, seed)
    if mult:
        w.p["dem_mult"] = mult
        days, _ = make_days(w, np.random.default_rng(20_000 + seed), RV.TEST)
    hard = group == "constraints"
    it = 100 if hard else 50
    res = {"nn_check": evaluate2(w, days, nn, hard),
           "nn_rollout4": evaluate2(w, days, RolloutPolicy(nn, 4), hard),
           "alns_from_rollout4": evaluate2(w, days, ALNSPolicy(1e9, seed=95_000 + seed, max_iters=it,
                                                               init=RolloutPolicy(nn, 4)), hard)}
    save(path, dict(group=group, key=key, seed=seed, results=res, runtime_s=time.time() - t0))
    print(f"{group} {key} s{seed} {time.time() - t0:.0f}s", flush=True)


def init(p):
    global DATA
    DATA = load(p)
    RV.DATA = DATA


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="results/rev_robust")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    jobs = [("constraints", k, s, a.out) for k in RV.CONSTRAINTS for s in RV.SEEDS]
    jobs += [("sens", k, s, a.out) for k in RV.SENS for s in RV.SEEDS]
    with Pool(a.workers, initializer=init, initargs=(a.data,)) as P:
        list(P.imap_unordered(job, jobs))
