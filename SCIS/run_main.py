"""Main experiment (Tables 4-7, Figures 6-9, 11). One JSON per seed; re-running skips finished seeds,
so an interrupted run resumes where it stopped.

    python run_main.py --data "Supply_chain_logisitcs_problem.xlsx" --seeds 0-29
"""
import argparse, json, os, time, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scs.data import load
from scs.env import World, execute, day_metrics
from scs.policies import Greedy, NN2Opt, Ratio2Opt, TabularQ, NeuralQ
from scs.ledger import Ledger, planned_loads

TRAIN_EPOCHS, TEST_EPOCHS = 800, 200
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4]
SENS_EPOCHS = 100


class Forecaster:
    def __init__(self, world):
        self.value = world.mean_dem.copy()
        self.a = world.p["ewma"]

    def update(self, day):
        ok = ~day["miss"]
        self.value[ok] = (1 - self.a) * self.value[ok] + self.a * day["obs"][ok]


def train(policy, world, rng, epochs, curve=None):
    fc = Forecaster(world)
    for e in range(epochs):
        if isinstance(policy, NeuralQ):
            policy.set_progress(e / epochs)
        day = world.draw_day(rng, forecast=fc.value)
        trips = policy.plan(world, day)
        out = execute(world, day, trips)
        policy.learn(world, out)
        fc.update(day)
        if curve is not None:
            m = day_metrics(world, day, trips, out)
            curve.append((m["cost"], m["ds"], m["cpu"]))
    policy.training = False
    if isinstance(policy, NeuralQ):
        policy.eps = 0.0


def make_days(world, rng, n, sigma=None, p_missing=None):
    fc = Forecaster(world)
    days, fc_err = [], []
    for _ in range(n):
        d = world.draw_day(rng, sigma=sigma, p_missing=p_missing, forecast=fc.value)
        fc_err.append(np.abs(fc.value - d["true"]).mean())
        fc.update(d)
        days.append(d)
    return days, float(np.mean(fc_err))


def evaluate(world, days, policy, use_ledger, ledger_kw=None):
    led = Ledger(**(ledger_kw or {})) if use_ledger else None
    fallback = Greedy()
    rows, rejected = [], 0
    for e, day in enumerate(days):
        t0 = time.perf_counter()
        trips = policy.plan(world, day)
        t_plan = 1000 * (time.perf_counter() - t0)
        t_val = t_cons = 0.0
        if led is not None:
            ok, t_val, t_cons = led.submit(world, day["obs"], e, planned_loads(world, day["obs"], trips))
            if not ok:
                rejected += 1
                trips = fallback.plan(world, day)
        out = execute(world, day, trips)
        m = day_metrics(world, day, trips, out)
        m.update(t_plan=t_plan, t_val=t_val, t_cons=t_cons, latency=t_plan + t_val + t_cons)
        rows.append(m)
    res = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    res["bc_overhead"] = (res["t_val"] + res["t_cons"]) / res["latency"] if res["latency"] else 0.0
    if led is not None:
        res["chain_intact"] = led.verify() == -1
        res["rejected"] = rejected
    return res


def run_seed(data, seed, out_dir, sens=True):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"seed_{seed:03d}.json")
    if os.path.exists(path):
        return
    t_start = time.time()
    rng_world = np.random.default_rng(10_000 + seed)
    world = World(data, rng_world)
    days, fc_mae = make_days(world, np.random.default_rng(20_000 + seed), TEST_EPOCHS)

    pol = {}
    curves = {}
    for name, cls, joint in (("tab_joint", TabularQ, True), ("tab_sep", TabularQ, False),
                             ("nn_joint", NeuralQ, True), ("nn_sep", NeuralQ, False)):
        rng = np.random.default_rng(30_000 + 100 * seed + len(pol))
        p = cls(joint, rng) if cls is TabularQ else cls(joint, rng, n_max=world.n)
        curves[name] = []
        train(p, world, np.random.default_rng(40_000 + seed), TRAIN_EPOCHS, curves[name])
        pol[name] = p

    configs = {
        "iot_only":     (Greedy(), False),
        "bc_only":      (Greedy(), True),
        "ai_only":      (pol["tab_joint"], False),
        "separate":     (pol["tab_sep"], True),
        "proposed":     (pol["tab_joint"], True),
        "nn2opt":       (NN2Opt(), False),
        "ratio2opt":    (Ratio2Opt(), False),
        "nn_separate":  (pol["nn_sep"], True),
        "nn_proposed":  (pol["nn_joint"], True),
    }
    res = {k: evaluate(world, days, p, led) for k, (p, led) in configs.items()}

    sens_res = {}
    if sens:
        for lvl in NOISE_LEVELS:
            sd, _ = make_days(world, np.random.default_rng(50_000 + seed), SENS_EPOCHS,
                              sigma=lvl, p_missing=lvl / 4)
            sens_res[str(lvl)] = {k: evaluate(world, sd, configs[k][0], False)
                                  for k in ("iot_only", "nn2opt", "ratio2opt", "proposed", "nn_proposed")}

    def binned(c, nb=10):
        a = np.array(c)
        return [a[i * len(a) // nb:(i + 1) * len(a) // nb].mean(0).tolist() for i in range(nb)]

    out = dict(seed=seed, results=res, sensitivity=sens_res,
               curves={k: binned(v) for k, v in curves.items()},
               q_table_size={k: pol[k].size() for k in ("tab_joint", "tab_sep")},
               nn_params=pol["nn_joint"].net.n_params(), forecast_mae=fc_mae,
               world=dict(n=world.n, K=world.K, Q=world.Q), runtime_s=time.time() - t_start)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, path)      # atomic: a power cut never leaves a half-written result
    print(f"seed {seed} done in {out['runtime_s']:.0f}s", flush=True)


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = map(int, part.split("-")); out += list(range(a, b + 1))
        else:
            out.append(int(part))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", default="0-29")
    ap.add_argument("--out", default="results/main")
    ap.add_argument("--sens-seeds", default="0-11", help="seeds that also run the sensitivity analysis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = load(a.data)
    sens = set(parse_seeds(a.sens_seeds))
    for s in parse_seeds(a.seeds):
        run_seed(data, s, a.out, sens=s in sens)
