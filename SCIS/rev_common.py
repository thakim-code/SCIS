"""Shared helpers for the first-round revision experiments."""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scs.env import World, execute, day_metrics, PlanState
from scs.policies import Greedy, NN2Opt, Ratio2Opt, TabularQ, NeuralQ, plan_day
from scs.ledger import Ledger, planned_loads, check_plan
from scs.revision import ALNSPolicy, ORToolsPolicy, Post2Opt, coverage_metrics, tw_two_opt
from run_main import Forecaster, make_days


def train_diag(policy, world, rng, epochs, val_days=None, every=80):
    """Identical to run_main.train (same RNG consumption) plus diagnostics: per-epoch training
    metrics, epsilon, and a greedy (no exploration, no learning) checkpoint on validation days."""
    fc = Forecaster(world)
    curve, ckpt = [], []
    for e in range(epochs):
        if isinstance(policy, NeuralQ):
            policy.set_progress(e / epochs)
        day = world.draw_day(rng, forecast=fc.value)
        trips = policy.plan(world, day)
        out = execute(world, day, trips)
        policy.learn(world, out)
        fc.update(day)
        m = day_metrics(world, day, trips, out)
        curve.append((m["cost"], m["ds"], m["cpu"], getattr(policy, "eps", 0.0)))
        if val_days is not None and (e + 1) % every == 0:
            policy.training = False
            eps = getattr(policy, "eps", None)
            if isinstance(policy, NeuralQ):
                policy.eps = 0.0
            vm = [day_metrics(world, d, t, execute(world, d, t))
                  for d in val_days for t in [policy.plan(world, d)]]
            ckpt.append(dict(epoch=e + 1, cpu=float(np.nanmean([x["cpu"] for x in vm])),
                             ds=float(np.mean([x["ds"] for x in vm])),
                             qsize=policy.size() if isinstance(policy, TabularQ) else None))
            policy.training = True
            if eps is not None:
                policy.eps = eps
    policy.training = False
    if isinstance(policy, NeuralQ):
        policy.eps = 0.0
    return curve, ckpt


class RatioCov:
    """Demand-aware greedy with a coverage weight lam: maximise (load/Q + lam)/(km + 1), then 2-opt."""

    def __init__(self, lam):
        self.lam = lam
        self.name = f"ratio_cov{lam}"

    def plan(self, world, day):
        def ch(ps, c):
            load = np.minimum(ps.day["obs"][c], ps.cap)
            return int(c[np.argmax((load / world.Q + self.lam) / (world.D[ps.cur, c + 1] + 1.0))]), None
        return [tw_two_opt(world, t) for t in plan_day(world, day, ch)[0]]


class RatioTW:
    """Demand-aware greedy + 2-opt that keeps hard time windows (2-opt move rejected if it breaks one)."""
    name = "ratio_tw"

    def plan(self, world, day):
        def ch(ps, c):
            load = np.minimum(ps.day["obs"][c], ps.cap)
            return int(c[np.argmax(load / (world.D[ps.cur, c + 1] + 1.0))]), None
        return [tw_two_opt(world, t) for t in plan_day(world, day, ch)[0]]


def repair(world, obs, plan):
    """Validate-and-repair fallback: drop phantom and repeated customers, clip loads to observed
    demand and remaining capacity, and truncate each trip to its longest prefix that returns within
    the shift (and meets hard due times). The result always passes check_plan."""
    seen, out = set(), []
    for k, (seq, loads) in enumerate(plan):
        cap, t, cur, s2, l2 = world.Qk[k], 0.0, 0, [], []
        for j, l in zip(seq, loads):
            if not (0 <= j < world.n) or j in seen:
                continue
            arr = t + world.D[cur, j + 1] / world.v
            if arr + world.svc_h + world.D[j + 1, 0] / world.v > world.shift + 1e-9:
                break
            if world.p.get("hard_tw") and arr > world.due[j] + 1e-9:
                break
            l = max(0.0, min(l, obs[j], cap))
            s2.append(j); l2.append(l); seen.add(j)
            cap -= l; t = arr + world.svc_h; cur = j + 1
        out.append((s2, l2))
    return out


def evaluate2(world, days, policy, use_ledger, fault_rate=0.0, fault_rng=None, corrupt=None,
              fallback_mode="dispatch"):
    """evaluate() from run_main plus coverage, emissions proxy, and router-fault handling.
    With fault_rate > 0 a share of plans is corrupted (one constraint violation each) before
    validation; without the ledger such plans are executed (phantom stops skipped) and counted."""
    led = Ledger() if use_ledger else None
    fallback = Greedy()
    rows, served, rejected, exec_invalid = [], [], 0, 0
    for e, day in enumerate(days):
        t0 = time.perf_counter()
        trips = policy.plan(world, day)
        t_plan = 1000 * (time.perf_counter() - t0)
        plan = planned_loads(world, day["obs"], trips)
        bad = False
        if fault_rate > 0 and fault_rng.random() < fault_rate:
            from run_ledger import CORRUPTIONS
            kind = CORRUPTIONS[int(fault_rng.integers(len(CORRUPTIONS)))]
            plan = corrupt(world, plan, kind, fault_rng)
            bad = True
        t_val = t_cons = 0.0
        if led is not None:
            ok, t_val, t_cons = led.submit(world, day["obs"], e, plan)
            if not ok:
                rejected += 1
                if fallback_mode == "repair":
                    plan = repair(world, day["obs"], plan)
                    assert check_plan(world, day["obs"], plan)
                    bad = False
                else:
                    trips = fallback.plan(world, day)
                    plan = None
        if plan is not None:
            if bad:
                exec_invalid += 1
            trips = [[j for j in s if 0 <= j < world.n] for s, _ in plan]
        out = execute(world, day, trips)
        m = day_metrics(world, day, trips, out)
        m.update(t_plan=t_plan, t_val=t_val, t_cons=t_cons, latency=t_plan + t_val + t_cons)
        rows.append(m)
        served.append(np.unique([s["j"] for tr in out for s in tr["stops"] if s["delivered"] > 0]))
    res = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    res["latency_p95"] = float(np.percentile([r["latency"] for r in rows], 95))
    res.update(coverage_metrics(world.n, served))
    if led is not None:
        res["chain_intact"] = led.verify() == -1
        res["rejected"] = rejected
    res["exec_invalid_share"] = exec_invalid / len(days)
    return res


def save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)
