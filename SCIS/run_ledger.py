"""Ledger experiments (new Section 6.E): (a) tamper detection, (b) invalid-decision injection with
faulty (always-approve) validators.

    python run_ledger.py --data "Supply_chain_logisitcs_problem.xlsx"
"""
import argparse, copy, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scs.data import load
from scs.env import World
from scs.policies import Greedy
from scs.ledger import Ledger, planned_loads

CORRUPTIONS = ("overload", "duplicate", "overtime", "phantom")


def corrupt(world, plan, kind, rng):
    """Return a copy of the plan that violates one shared constraint."""
    plan = copy.deepcopy(plan)
    k = int(rng.integers(len(plan)))
    seq, loads = list(plan[k][0]), list(plan[k][1])
    used = {j for s, _ in plan for j in s}
    free = [j for j in range(world.n) if j not in used]
    if kind == "overload":                      # total load exceeds vehicle capacity
        if not seq:
            seq, loads = [free[0]], [0.0]
        loads[-1] += 0.3 * world.Q
    elif kind == "duplicate":                   # a customer visited twice in the day
        seq.append(next(iter(used)) if used else 0); loads.append(0.0)
    elif kind == "overtime":                    # route cannot return within the shift
        far = sorted(free, key=lambda j: -world.D[0, j + 1])[:12]
        seq += far; loads += [0.0] * len(far)
    elif kind == "phantom":                     # non-existent customer
        seq.append(world.n + 7); loads.append(0.0)
    plan[k] = (seq, loads)
    return plan


def tamper(led, rng, trials=1000):
    res = dict(edit_no_rehash=0, edit_rehash_block=0, edit_rehash_suffix_single_copy=0)
    n = len(led.chain)
    for _ in range(trials):
        i = int(rng.integers(1, n - 1))           # any non-genesis, non-tip block
        for mode in res:
            ch = copy.deepcopy(led.chain)
            rec = ch[i]["body"]["record"]
            if rec["plan"] and rec["plan"][0][1]:
                rec["plan"][0][1][0] += 1.0
            else:
                rec["epoch"] += 1
            if mode != "edit_no_rehash":
                ch[i]["hash"] = Ledger._hash(ch[i]["body"])
            if mode == "edit_rehash_suffix_single_copy":
                for m in range(i + 1, n):
                    ch[m]["body"]["prev"] = ch[m - 1]["hash"]
                    ch[m]["hash"] = Ledger._hash(ch[m]["body"])
            l2 = Ledger(); l2.chain = ch
            res[mode] += l2.verify() != -1
    out = {k: v / trials for k, v in res.items()}
    # replicated ledger: 5 validator copies; attacker rewrites f copies consistently;
    # a node detects divergence when the tip hash of an honest copy differs from its own.
    out["replicated_suffix_rewrite_detected_by_honest_copy"] = {str(f): float(f < 5) for f in range(1, 6)}
    out["replicated_suffix_rewrite_accepted_by_majority"] = {str(f): float(f >= 3) for f in range(1, 6)}
    return out


def injection(world, data_days, faulty, rng, rate=0.10):
    led = Ledger(faulty=faulty)
    g = Greedy()
    inv_total = inv_acc = val_total = val_rej = 0
    by_kind = {k: [0, 0] for k in CORRUPTIONS}
    excluded_faulty = excluded_honest = 0
    for e, day in enumerate(data_days):
        plan = planned_loads(world, day["obs"], g.plan(world, day))
        bad = rng.random() < rate
        if bad:
            kind = CORRUPTIONS[int(rng.integers(len(CORRUPTIONS)))]
            plan = corrupt(world, plan, kind, rng)
        ok, _, _ = led.submit(world, day["obs"], e, plan)
        if bad:
            inv_total += 1; inv_acc += ok
            by_kind[kind][0] += 1; by_kind[kind][1] += ok
        else:
            val_total += 1; val_rej += (not ok)
        excluded_faulty += int((led.trust[led.faulty] < led.tau).sum())
        excluded_honest += int((led.trust[~led.faulty] < led.tau).sum())
    T = len(data_days)
    return dict(faulty=faulty, invalid_accept_rate=inv_acc / max(inv_total, 1),
                valid_reject_rate=val_rej / max(val_total, 1),
                n_invalid=inv_total,
                by_kind={k: (v[1] / v[0] if v[0] else None) for k, v in by_kind.items()},
                final_trust_faulty=float(led.trust[led.faulty].mean()) if faulty else None,
                final_trust_honest=float(led.trust[~led.faulty].mean()) if faulty < 5 else None,
                share_faulty_excluded=excluded_faulty / (T * faulty) if faulty else None,
                share_honest_excluded=excluded_honest / (T * (5 - faulty)) if faulty < 5 else None,
                chain_intact=led.verify() == -1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--out", default="results/ledger.json")
    a = ap.parse_args()
    data = load(a.data)
    tam, inj = [], {f: [] for f in range(4)}
    for s in range(a.seeds):
        world = World(data, np.random.default_rng(10_000 + s))
        rng = np.random.default_rng(60_000 + s)
        days = [world.draw_day(rng) for _ in range(a.epochs)]
        led = Ledger()
        g = Greedy()
        for e, d in enumerate(days[:200]):
            led.submit(world, d["obs"], e, planned_loads(world, d["obs"], g.plan(world, d)))
        tam.append(tamper(led, np.random.default_rng(70_000 + s), trials=200))
        for f in range(4):
            inj[f].append(injection(world, days, f, np.random.default_rng(80_000 + 10 * s + f)))
        print("ledger seed", s, flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(tamper=tam, injection={str(k): v for k, v in inj.items()}), open(a.out, "w"))
