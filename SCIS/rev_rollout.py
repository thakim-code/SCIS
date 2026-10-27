"""Revision (learned-engine improvement): decision-time rollout of the neural router and ALNS seeded by it.
Same worlds, held-out days and trained neural router (same RNG streams) as rev_main.py.
    python rev_rollout.py --data DATA.xlsx --seeds 0-29 --workers 2
"""
import argparse, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_common import *
from run_main import TRAIN_EPOCHS, TEST_EPOCHS, parse_seeds, train
from scs.data import load
from scs.revision import RolloutPolicy

DATA = None


def job(args):
    seed, out = args
    path = os.path.join(out, f"seed_{seed:03d}.json")
    if os.path.exists(path):
        return
    t0 = time.time()
    w = World(DATA, np.random.default_rng(10_000 + seed))
    days, _ = make_days(w, np.random.default_rng(20_000 + seed), TEST_EPOCHS)
    nn = NeuralQ(True, np.random.default_rng(30_000 + 100 * seed + 2), n_max=w.n)
    train(nn, w, np.random.default_rng(40_000 + seed), TRAIN_EPOCHS)
    res = {"nn_rollout4": evaluate2(w, days, RolloutPolicy(nn, 4), True),
           "nn_rollout8": evaluate2(w, days, RolloutPolicy(nn, 8), True),
           "alns_from_rollout4": evaluate2(w, days, ALNSPolicy(1e9, seed=95_000 + seed, max_iters=150,
                                                               init=RolloutPolicy(nn, 4)), True),
           "nn_check": evaluate2(w, days, nn, True)}
    save(path, dict(seed=seed, results=res, runtime_s=time.time() - t0))
    print(f"rollout seed {seed} {time.time() - t0:.0f}s", flush=True)


def init(p):
    global DATA
    DATA = load(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--seeds", default="0-29")
    ap.add_argument("--out", default="results/rev_rollout"); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    with Pool(a.workers, initializer=init, initargs=(a.data,)) as P:
        list(P.imap_unordered(job, [(s, a.out) for s in parse_seeds(a.seeds)]))
