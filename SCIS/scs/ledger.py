"""Lightweight permissioned ledger for decision validation (a simulation, not a deployed chain).

Each day's routing plan is a decision record. N validators independently check it against the
shared constraints; a majority vote among validators whose trust >= tau decides acceptance;
accepted and rejected decisions are both appended to a SHA-256 hash chain. Timings are real
wall-clock measurements of this implementation.
"""
import hashlib
import json
import time
import numpy as np


def check_plan(world, obs, trips):
    """Honest validator: capacity, shift, uniqueness, index validity, and load consistency."""
    seen = set()
    for k, (seq, loads) in enumerate(trips):
        Qk = world.Qk[k] if k < len(world.Qk) else world.Q
        if len(seq) != len(loads):
            return False
        if sum(loads) > Qk * (1 + 1e-9):
            return False
        t, cur, cap = 0.0, 0, Qk
        for j, l in zip(seq, loads):
            if not (0 <= j < world.n) or j in seen:
                return False
            seen.add(j)
            if l < -1e-9 or l > obs[j] + 1e-6 or l > cap + 1e-6:
                return False
            cap -= l
            if world.p.get("hard_tw") and t + world.D[cur, j + 1] / world.v > world.due[j] + 1e-9:
                return False
            t += world.D[cur, j + 1] / world.v + world.svc_h
            cur = j + 1
        if t + world.D[cur, 0] / world.v > world.shift + 1e-9:
            return False
    return True


def planned_loads(world, obs, trips):
    out = []
    for k, seq in enumerate(trips):
        cap, loads = world.Qk[k], []
        for j in seq:
            l = min(obs[j], cap)
            cap -= l
            loads.append(float(l))
        out.append((list(map(int, seq)), loads))
    return out


class Ledger:
    def __init__(self, n_validators=5, faulty=0, beta=0.9, tau=0.7):
        self.n, self.beta, self.tau = n_validators, beta, tau
        self.faulty = np.zeros(n_validators, bool)
        self.faulty[:faulty] = True
        self.trust = np.ones(n_validators)
        self.chain = [self._block(0, "0" * 64, {"genesis": True}, True)]
        self.trust_hist = []

    @staticmethod
    def _hash(body):
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _block(self, idx, prev, record, accepted):
        body = dict(index=idx, prev=prev, record=record, accepted=bool(accepted))
        return dict(body=body, hash=self._hash(body))

    def submit(self, world, obs, epoch, plan):
        """plan: list of (seq, loads). Returns (accepted, t_validation_ms, t_consensus_ms)."""
        t0 = time.perf_counter()
        votes = np.array([True if f else check_plan(world, obs, plan) for f in self.faulty])
        t1 = time.perf_counter()
        elig = self.trust >= self.tau
        if not elig.any():
            elig = np.ones(self.n, bool)
        accepted = votes[elig].sum() * 2 > elig.sum()
        self.trust = self.beta * self.trust + (1 - self.beta) * (votes == accepted)
        self.trust_hist.append(self.trust.copy())
        rec = dict(epoch=int(epoch), plan=[[s, [round(x, 3) for x in l]] for s, l in plan])
        blk = self._block(len(self.chain), self.chain[-1]["hash"], rec, accepted)
        self.chain.append(blk)
        t2 = time.perf_counter()
        return bool(accepted), 1000 * (t1 - t0), 1000 * (t2 - t1)

    def verify(self):
        """Return index of first invalid block, or -1 if the chain is intact."""
        for i, b in enumerate(self.chain):
            if self._hash(b["body"]) != b["hash"]:
                return i
            if i > 0 and b["body"]["prev"] != self.chain[i - 1]["hash"]:
                return i
        return -1
