"""Minor revision: adds the OR-Tools GLS baseline (1 s/day) to result files of rev_main.py that do not
yet contain it (the first-round files ran OR-Tools on seeds 0-9 only). Worlds and test days are
regenerated from the same seeds as rev_main.py, so OR-Tools is evaluated on the identical instances,
with the same evaluator (execution against true demand, real weight-banded freight) as every engine.
OR-Tools stops on a wall-clock limit, so its plans vary slightly with machine speed.

    python rev_ortools.py --data DATA.xlsx --seeds 10-29 --workers 2
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_common import *
from run_main import TEST_EPOCHS, parse_seeds
from scs.data import load

DATA = None


def job(args):
    seed, out_dir = args
    path = os.path.join(out_dir, f"seed_{seed:03d}.json")
    d = json.load(open(path))
    if "ortools" in d["results"]:
        return
    t = time.time()
    world = World(DATA, np.random.default_rng(10_000 + seed))
    days, _ = make_days(world, np.random.default_rng(20_000 + seed), TEST_EPOCHS)
    d["results"]["ortools"] = evaluate2(world, days, ORToolsPolicy(1.0), True)
    save(path, d)
    print(f"rev_ortools seed {seed} done in {time.time() - t:.0f}s", flush=True)


def init(path):
    global DATA
    DATA = load(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", default="10-29")
    ap.add_argument("--out", default="results/rev_main")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    with Pool(a.workers, initializer=init, initargs=(a.data,)) as P:
        list(P.imap_unordered(job, [(s, a.out) for s in parse_seeds(a.seeds)]))
