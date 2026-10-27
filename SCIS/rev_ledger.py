"""Revision: ledger experiments.

(1) BFT scaling. Discrete-event simulation of the PBFT normal case (pre-prepare, prepare, commit;
    quorum 2f+1, f = floor((N-1)/3)) for N = 4..100 validators, with per-node sequential CPU queues
    and per-node uplink serialisation. CPU costs are MEASURED on this machine: Ed25519 sign/verify
    (cryptography), SHA-256 block hashing and the constraint check of real daily plans. Network
    one-way delays are log-normal (LAN median 0.2 ms; regional WAN 20 ms; global WAN 80 ms; sigma 0.3)
    with 1 Gbit/s, 100 Mbit/s and 50 Mbit/s uplinks. Runs with all validators honest and with f
    silent (crashed or withholding) validators.
(2) Evidence-based trust. Validators attach a verifiable violation witness (constraint, trip, stop)
    to a reject vote; any witness that re-checks as true rejects the plan regardless of the vote
    count, and trust is updated against the VERIFIED outcome instead of agreement with the majority.
    Compared with the original majority-agreement ledger for 0-4 of 5 faulty validators that
    approve everything or reject everything.

    python rev_ledger.py --data DATA.xlsx
"""
import argparse, heapq, json, os, sys, time, copy
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scs.data import load
from scs.env import World
from scs.policies import Greedy
from scs.ledger import Ledger, planned_loads, check_plan
from run_ledger import corrupt, CORRUPTIONS
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import hashlib


# ------------------------------------------------------------------ measured CPU costs
def measure_costs(world, plans, reps=2000):
    key = Ed25519PrivateKey.generate(); pub = key.public_key()
    msg = os.urandom(120)
    t = time.perf_counter()
    for _ in range(reps):
        sig = key.sign(msg)
    t_sign = (time.perf_counter() - t) / reps
    t = time.perf_counter()
    for _ in range(reps):
        pub.verify(sig, msg)
    t_ver = (time.perf_counter() - t) / reps
    blobs = [json.dumps(p).encode() for p in plans]
    t = time.perf_counter()
    for _ in range(reps // 10):
        for b in blobs:
            hashlib.sha256(b).hexdigest()
    t_hash = (time.perf_counter() - t) / (reps // 10 * len(blobs))
    t = time.perf_counter()
    for _ in range(20):
        for p in plans:
            _chk(world, p)
    t_chk = (time.perf_counter() - t) / (20 * len(plans))
    size = float(np.mean([len(b) for b in blobs])) + 64 + 64     # plan + header + signature
    return dict(sign_ms=1e3 * t_sign, verify_ms=1e3 * t_ver, hash_ms=1e3 * t_hash,
                check_ms=1e3 * t_chk, plan_bytes=size, vote_bytes=32 + 64 + 48)


def _chk(world, plan):
    obs = np.full(world.n, 1e12)
    return check_plan(world, obs, plan)


PROFILES = {"LAN": (0.2, 1e9), "WAN-regional": (20.0, 1e8), "WAN-global": (80.0, 5e7)}


def pbft_round(N, costs, prof, rng, silent=0):
    """Return (latency_ms to 2f+1 commits, messages, bytes, max per-node CPU ms)."""
    med, bw = PROFILES[prof]
    f = (N - 1) // 3
    q = 2 * f + 1
    alive = np.ones(N, bool)
    if silent:
        alive[N - silent:] = False          # never the primary (node 0)
    lat = lambda: rng.lognormal(np.log(med), 0.3)
    ser = lambda b: 1e3 * 8 * b / bw
    cpu_free = np.zeros(N); cpu_used = np.zeros(N); up_free = np.zeros(N)
    ev = []  # (time, node, kind, sender)
    msgs = 0; nbytes = 0.0
    # primary: hash + sign + check, then send pre-prepare to all
    t0 = costs["hash_ms"] + costs["sign_ms"] + costs["check_ms"]
    cpu_used[0] += t0; cpu_free[0] = t0
    for i in range(1, N):
        up_free[0] = max(up_free[0], t0) + ser(costs["plan_bytes"])
        heapq.heappush(ev, (up_free[0] + lat(), i, "pp", 0)); msgs += 1; nbytes += costs["plan_bytes"]
    prep = [set() for _ in range(N)]; comm = [set() for _ in range(N)]
    sent_prep = np.zeros(N, bool); sent_comm = np.zeros(N, bool); done = np.full(N, np.inf)
    sent_prep[0] = True; prep[0].add(0)

    def bcast(i, t, kind):
        nonlocal msgs, nbytes
        for j in range(N):
            if j == i:
                continue
            up_free[i] = max(up_free[i], t) + ser(costs["vote_bytes"])
            heapq.heappush(ev, (up_free[i] + lat(), j, kind, i)); msgs += 1; nbytes += costs["vote_bytes"]

    while ev:
        t, i, kind, s = heapq.heappop(ev)
        if not alive[i]:
            continue
        work = costs["verify_ms"] + (costs["check_ms"] + costs["hash_ms"] if kind == "pp" else 0.0)
        start = max(t, cpu_free[i]); cpu_free[i] = start + work; cpu_used[i] += work
        now = cpu_free[i]
        if kind == "pp" and not sent_prep[i]:
            sent_prep[i] = True; prep[i].add(i); prep[i].add(0)   # pre-prepare counts as the primary's prepare
            cpu_free[i] += costs["sign_ms"]; cpu_used[i] += costs["sign_ms"]
            bcast(i, cpu_free[i], "prep")
        elif kind == "prep":
            prep[i].add(s)
        elif kind == "com":
            comm[i].add(s)
        if sent_prep[i] and not sent_comm[i] and len(prep[i]) >= q:
            sent_comm[i] = True; comm[i].add(i)
            cpu_free[i] += costs["sign_ms"]; cpu_used[i] += costs["sign_ms"]
            bcast(i, cpu_free[i], "com")
        if sent_comm[i] and done[i] == np.inf and len(comm[i]) >= q:
            done[i] = cpu_free[i]
    fin = np.sort(done[alive])
    latency = fin[q - 1] if np.isfinite(fin[q - 1]) else np.inf
    return float(latency), msgs, nbytes, float(cpu_used.max())


# ------------------------------------------------------------------ evidence-based trust
def violations_detail(world, obs, plan):
    """All violated constraints, each as a witness tuple (kind, trip, stop)."""
    out, seen = [], set()
    for k, (seq, loads) in enumerate(plan):
        Qk = world.Qk[k]
        if sum(loads) > Qk * (1 + 1e-9):
            out.append(("capacity", k, -1))
        t, cur, cap = 0.0, 0, Qk
        for p, (j, l) in enumerate(zip(seq, loads)):
            if not (0 <= j < world.n):
                out.append(("phantom", k, p)); continue
            if j in seen:
                out.append(("duplicate", k, p))
            seen.add(j)
            if l > obs[j] + 1e-6 or l > cap + 1e-6:
                out.append(("load", k, p))
            cap -= l
            t += world.D[cur, j + 1] / world.v + world.svc_h; cur = j + 1
        if t + world.D[cur, 0] / world.v > world.shift + 1e-9:
            out.append(("shift", k, -1))
    return out


class EvidenceLedger(Ledger):
    """Majority vote among trusted validators, overridden by any VERIFIED violation witness;
    trust updated against the verified outcome."""

    def __init__(self, faulty=0, behaviour="approve", **kw):
        super().__init__(faulty=faulty, **kw)
        self.behaviour = behaviour

    def submit(self, world, obs, epoch, plan):
        t0 = time.perf_counter()
        viol = violations_detail(world, obs, plan)
        votes, wits = [], []
        for f in self.faulty:
            if f and self.behaviour == "approve":
                votes.append(True); wits.append(None)
            elif f and self.behaviour == "reject":
                votes.append(False); wits.append(("capacity", 0, -1))      # fabricated witness
            else:
                votes.append(not viol); wits.append(viol[0] if viol else None)
        votes = np.array(votes)
        t1 = time.perf_counter()
        verified = [w is not None and w in viol for w in wits]        # re-check every witness
        if any(verified):
            accepted, truth = False, False
        else:
            elig = self.trust >= self.tau
            if not elig.any():
                elig = np.ones(self.n, bool)
            # a reject vote without a verifiable witness carries no weight
            eff = np.where(votes | np.array(verified), votes, True)
            accepted = eff[elig].sum() * 2 > elig.sum()
            truth = True
        correct = votes == truth
        self.trust = self.beta * self.trust + (1 - self.beta) * correct
        self.trust_hist.append(self.trust.copy())
        rec = dict(epoch=int(epoch), plan=[[s, [round(x, 3) for x in l]] for s, l in plan])
        self.chain.append(self._block(len(self.chain), self.chain[-1]["hash"], rec, accepted))
        t2 = time.perf_counter()
        return bool(accepted), 1000 * (t1 - t0), 1000 * (t2 - t1)


class MajorityRejectLedger(Ledger):
    """Original ledger, but faulty validators reject everything (for the 'reject' behaviour)."""

    def submit(self, world, obs, epoch, plan):
        t0 = time.perf_counter()
        votes = np.array([False if f else check_plan(world, obs, plan) for f in self.faulty])
        t1 = time.perf_counter()
        elig = self.trust >= self.tau
        if not elig.any():
            elig = np.ones(self.n, bool)
        accepted = votes[elig].sum() * 2 > elig.sum()
        self.trust = self.beta * self.trust + (1 - self.beta) * (votes == accepted)
        self.trust_hist.append(self.trust.copy())
        rec = dict(epoch=int(epoch), plan=[[s, [round(x, 3) for x in l]] for s, l in plan])
        self.chain.append(self._block(len(self.chain), self.chain[-1]["hash"], rec, accepted))
        return bool(accepted), 1000 * (t1 - t0), 0.0


def injection(world, days, led, rng, rate=0.10):
    g = Greedy()
    inv = inv_acc = val = val_rej = 0
    exf = exh = 0
    for e, day in enumerate(days):
        plan = planned_loads(world, day["obs"], g.plan(world, day))
        bad = rng.random() < rate
        if bad:
            plan = corrupt(world, plan, CORRUPTIONS[int(rng.integers(len(CORRUPTIONS)))], rng)
        ok, _, _ = led.submit(world, day["obs"], e, plan)
        if bad:
            inv += 1; inv_acc += ok
        else:
            val += 1; val_rej += (not ok)
        exf += int((led.trust[led.faulty] < led.tau).sum()); exh += int((led.trust[~led.faulty] < led.tau).sum())
    nf = int(led.faulty.sum()); T = len(days)
    return dict(invalid_accept=inv_acc / max(inv, 1), valid_reject=val_rej / max(val, 1),
                trust_faulty=float(led.trust[led.faulty].mean()) if nf else None,
                trust_honest=float(led.trust[~led.faulty].mean()) if nf < 5 else None,
                faulty_excluded=exf / (T * nf) if nf else None,
                honest_excluded=exh / (T * (5 - nf)) if nf < 5 else None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/rev_ledger.json")
    ap.add_argument("--seeds", type=int, default=30)
    a = ap.parse_args()
    data = load(a.data)
    out = {}
    # (1) BFT scaling
    scal = {}
    for n_c in (41, 160):
        w = World(data, np.random.default_rng(10_000), n_customers=None if n_c == 41 else n_c)
        rng = np.random.default_rng(60_000)
        plans = [planned_loads(w, d["obs"], Greedy().plan(w, d)) for d in [w.draw_day(rng) for _ in range(30)]]
        costs = measure_costs(w, plans)
        scal[str(n_c)] = dict(costs=costs, runs={})
        for prof in PROFILES:
            for N in (4, 7, 10, 16, 31, 64, 100):
                for silent in (0, (N - 1) // 3):
                    r = np.random.default_rng(61_000 + N)
                    rows = [pbft_round(N, costs, prof, r, silent) for _ in range(100)]
                    L = np.array([x[0] for x in rows])
                    scal[str(n_c)]["runs"][f"{prof}|{N}|{silent}"] = dict(
                        lat_med=float(np.median(L)), lat_p95=float(np.percentile(L, 95)),
                        msgs=float(np.mean([x[1] for x in rows])), kbytes=float(np.mean([x[2] for x in rows])) / 1e3,
                        cpu_max_ms=float(np.mean([x[3] for x in rows])))
        print("scaling", n_c, costs, flush=True)
    out["bft_scaling"] = scal
    # (2) evidence-based trust vs majority-agreement trust
    inj = {}
    for s in range(a.seeds):
        w = World(data, np.random.default_rng(10_000 + s))
        rng = np.random.default_rng(60_000 + s)
        days = [w.draw_day(rng) for _ in range(500)]
        for beh in ("approve", "reject"):
            for f in range(5):
                for mech in ("majority", "evidence"):
                    if mech == "majority":
                        led = Ledger(faulty=f) if beh == "approve" else MajorityRejectLedger(faulty=f)
                    else:
                        led = EvidenceLedger(faulty=f, behaviour=beh)
                    r = injection(w, days, led, np.random.default_rng(80_000 + 10 * s + f))
                    inj.setdefault(f"{mech}|{beh}|{f}", []).append(r)
        print("trust seed", s, flush=True)
    out["trust"] = inj
    json.dump(out, open(a.out, "w"))
