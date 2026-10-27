"""Revision, main experiment: the 30-seed protocol of run_main.py (identical worlds, days, training
streams and therefore identical learned policies) extended with
  * stronger baselines: ALNS on the exact objective (150 iterations/day) and OR-Tools GLS (1 s/day,
    all seeds), and the learned router followed by 2-opt;
  * training diagnostics (TD loss, mean Q, greedy validation checkpoints);
  * coverage/fairness and emissions-proxy (vehicle-km per 1000 units) metrics for every configuration;
  * router-fault experiment: corrupted learned plans with and without the validation layer.

    python rev_main.py --data DATA.xlsx --seeds 0-29 --workers 2
"""
import argparse, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_common import *
from run_main import TRAIN_EPOCHS, TEST_EPOCHS, parse_seeds
from run_ledger import corrupt
from scs.data import load

DATA = None
FAULT_RATES = [0.01, 0.05, 0.10, 0.20]


def job(args):
    seed, out_dir = args
    path = os.path.join(out_dir, f"seed_{seed:03d}.json")
    if os.path.exists(path):
        return
    t_start = time.time()
    world = World(DATA, np.random.default_rng(10_000 + seed))
    days, _ = make_days(world, np.random.default_rng(20_000 + seed), TEST_EPOCHS)
    val_days, _ = make_days(world, np.random.default_rng(25_000 + seed), 20)
    pol, diag = {}, {}
    for name, cls, joint in (("tab_joint", TabularQ, True), ("tab_sep", TabularQ, False),
                             ("nn_joint", NeuralQ, True), ("nn_sep", NeuralQ, False)):
        rng = np.random.default_rng(30_000 + 100 * seed + len(pol))
        p = cls(joint, rng) if cls is TabularQ else cls(joint, rng, n_max=world.n)
        curve, ckpt = train_diag(p, world, np.random.default_rng(40_000 + seed), TRAIN_EPOCHS, val_days)
        a = np.array(curve)
        diag[name] = dict(ckpt=ckpt, eps=[float(a[i * 80:(i + 1) * 80, 3].mean()) for i in range(10)])
        if isinstance(p, NeuralQ):
            td = np.array(p.td_log)
            nb = 20
            diag[name]["td_loss"] = [float(td[i * len(td) // nb:(i + 1) * len(td) // nb, 0].mean()) for i in range(nb)]
            diag[name]["mean_q"] = [float(td[i * len(td) // nb:(i + 1) * len(td) // nb, 1].mean()) for i in range(nb)]
            diag[name]["n_updates"] = len(td)
        pol[name] = p
    configs = {
        "iot_only": (Greedy(), False), "bc_only": (Greedy(), True),
        "ai_only": (pol["tab_joint"], False), "separate": (pol["tab_sep"], True),
        "proposed": (pol["tab_joint"], True), "nn2opt": (NN2Opt(), False),
        "ratio2opt": (Ratio2Opt(), False), "nn_separate": (pol["nn_sep"], True),
        "nn_proposed": (pol["nn_joint"], True),
        "nn_proposed_2opt": (Post2Opt(pol["nn_joint"]), True),
        "alns": (ALNSPolicy(1e9, seed=95_000 + seed, max_iters=150), True),
    }
    configs["ortools"] = (ORToolsPolicy(1.0), True)
    res = {k: evaluate2(world, days, p, led) for k, (p, led) in configs.items()}
    faults = {}
    for fr in FAULT_RATES:
        for mode, led, fb in (("noledger", False, "dispatch"), ("ledger_dispatch", True, "dispatch"),
                              ("ledger_repair", True, "repair")):
            faults[f"{fr}_{mode}"] = evaluate2(world, days, pol["nn_joint"], led, fault_rate=fr,
                                               fault_rng=np.random.default_rng(96_000 + seed + int(1000 * fr)),
                                               corrupt=corrupt, fallback_mode=fb)
    out = dict(seed=seed, results=res, faults=faults, diag=diag,
               alns_iters=150, runtime_s=time.time() - t_start)
    save(path, out)
    print(f"rev_main seed {seed} done in {out['runtime_s']:.0f}s", flush=True)


def init(path):
    global DATA
    DATA = load(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", default="0-29")
    ap.add_argument("--out", default="results/rev_main")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with Pool(a.workers, initializer=init, initargs=(a.data,)) as P:
        list(P.imap_unordered(job, [(s, a.out) for s in parse_seeds(a.seeds)]))
