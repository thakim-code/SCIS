"""Scaling experiment: tabular vs neural router on networks of 10-160 customers bootstrapped from the
real customer pool (fleet scaled proportionally). One JSON per (n, seed); resumable.

    python run_scaling.py --data "Supply_chain_logisitcs_problem.xlsx"
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scs.data import load
from scs.env import World
from scs.policies import Ratio2Opt, TabularQ, NeuralQ
from run_main import train, make_days, evaluate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sizes", default="10,20,40,80,160")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--out", default="results/scaling")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = load(a.data)
    for n in map(int, a.sizes.split(",")):
        for s in range(a.seeds):
            path = os.path.join(a.out, f"n{n:03d}_s{s}.json")
            if os.path.exists(path):
                continue
            world = World(data, np.random.default_rng(90_000 + 100 * n + s), n_customers=n)
            days, _ = make_days(world, np.random.default_rng(91_000 + 100 * n + s), a.test)
            tab = TabularQ(True, np.random.default_rng(92_000 + 100 * n + s))
            nn = NeuralQ(True, np.random.default_rng(93_000 + 100 * n + s), n_max=world.n)
            for p in (tab, nn):
                train(p, world, np.random.default_rng(94_000 + 100 * n + s), a.train)
            res = {k: evaluate(world, days, p, False) for k, p in
                   (("ratio2opt", Ratio2Opt()), ("tabular", tab), ("neural", nn))}
            out = dict(n=n, seed=s, K=world.K, q_table_size=tab.size(), nn_params=nn.net.n_params(), results=res)
            json.dump(out, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
            print(f"n={n} seed={s} q={tab.size()} cpu tab/nn/ratio="
                  f"{res['tabular']['cpu']:.3f}/{res['neural']['cpu']:.3f}/{res['ratio2opt']['cpu']:.3f}", flush=True)
