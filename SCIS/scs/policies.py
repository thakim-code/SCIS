"""Routing policies.

All policies build each vehicle trip one stop at a time from a PlanState, choosing among the
customers that are unvisited, have positive observed demand, and can be reached and returned
from within the shift. A trip ends when capacity is exhausted or no customer is feasible.
"""
import numpy as np
from .env import PlanState, two_opt


def plan_day(world, day, choose):
    visited = np.zeros(world.n, bool)
    trips, traces = [], []
    for k in range(world.K):
        ps = PlanState(world, day, k, visited)
        trace = []
        while True:
            c = ps.feasible()
            if len(c) == 0:
                break
            j, info = choose(ps, c)
            trace.append(info)
            ps.step(j)
        trips.append(ps.seq)
        traces.append(trace)
    return trips, traces


# ------------------------------------------------------------------ heuristics
def nearest(ps, c):
    return int(c[np.argmin(ps.w.D[ps.cur, c + 1])]), None


class Greedy:
    """Reactive dispatch heuristic (IoT-only / blockchain-only): nearest feasible customer."""
    name = "greedy"

    def plan(self, world, day):
        return plan_day(world, day, nearest)[0]


class NN2Opt:
    """Classical reference: nearest-neighbour construction + 2-opt on each trip."""
    name = "nn2opt"

    def plan(self, world, day):
        trips = plan_day(world, day, nearest)[0]
        return [two_opt(world, t) for t in trips]


def ratio(ps, c):
    """Demand-aware greedy: maximise deliverable load per km to reach the customer."""
    load = np.minimum(ps.day["obs"][c], ps.cap)
    return int(c[np.argmax(load / (ps.w.D[ps.cur, c + 1] + 1.0))]), None


class Ratio2Opt:
    """Demand-aware classical reference: load-per-distance greedy construction + 2-opt."""
    name = "ratio2opt"

    def plan(self, world, day):
        trips = plan_day(world, day, ratio)[0]
        return [two_opt(world, t) for t in trips]


# ------------------------------------------------------------------ reward
REWARD = dict(a_cost=0.1, a_delay=0.5, a_dem=3.0)


def rewards(world, outcome, joint, rw=REWARD):
    """Per-stop rewards from the executed outcome.
    joint=True : r = -a_cost*dcost - a_delay*delay + a_dem*delivered/Q   (integrated)
    joint=False: r = -a_cost*dcost                                       (separate, cost only)"""
    rs = []
    for tr in outcome:
        r = []
        for s in tr["stops"]:
            v = -rw["a_cost"] * s["dcost"]
            if joint:
                v += -rw["a_delay"] * s["delay"] + rw["a_dem"] * s["delivered"] / world.Q
                v += rw.get("a_cov", 0.0)          # revision: optional per-stop coverage bonus
            r.append(v)
        rs.append(r)
    return rs


# ------------------------------------------------------------------ tabular Q-learning
class TabularQ:
    """Tabular Q-learning. State = (current node, 5-bin fraction of remaining observed demand)
    for the integrated variant; (current node, 5-bin fraction of customers still unvisited) for
    the separate variant, which receives no demand information. Action = next customer."""

    def __init__(self, joint, rng, alpha=0.1, gamma=0.9, eps=0.1, bins=5):
        self.joint, self.rng = joint, rng
        self.alpha, self.gamma, self.eps, self.bins = alpha, gamma, eps, bins
        self.Q = {}
        self.name = "tabular_" + ("joint" if joint else "separate")
        self.training = True

    def state(self, ps):
        if self.joint:
            obs = ps.day["obs"]
            tot = obs.sum()
            f = obs[~ps.visited].sum() / tot if tot > 0 else 0.0
        else:
            f = (~ps.visited).mean()
        return (ps.cur, min(self.bins - 1, int(f * self.bins)))

    def q(self, s, a):
        return self.Q.get((s, int(a)), 0.0)

    def choose(self, ps, c):
        s = self.state(ps)
        if self.training and self.rng.random() < self.eps:
            j = int(self.rng.choice(c))
        else:
            qs = np.array([self.q(s, a) for a in c])
            best = np.flatnonzero(qs == qs.max())
            # random tie-breaking while training; deterministic (first maximum) at test time
            j = int(c[self.rng.choice(best)]) if self.training else int(c[best[0]])
        return j, (s, j, c)

    def plan(self, world, day):
        trips, self._traces = plan_day(world, day, self.choose)
        return trips

    def learn(self, world, outcome, traces=None):
        traces = traces or self._traces
        rs = rewards(world, outcome, self.joint)
        for trace, r in zip(traces, rs):
            for i, (s, a, c) in enumerate(trace):
                if i + 1 < len(trace):
                    s2, _, c2 = trace[i + 1]
                    nxt = max(self.q(s2, b) for b in c2)
                else:
                    nxt = 0.0
                old = self.q(s, a)
                self.Q[(s, a)] = old + self.alpha * (r[i] + self.gamma * nxt - old)

    def size(self):
        return len(self.Q)


# ------------------------------------------------------------------ neural candidate-scoring Q
N_FEAT = 10


def cand_features(ps, c, joint):
    """Per-candidate features; the candidate set itself tells the network which customers remain."""
    w, day = ps.w, ps.day
    obs = day["obs"]
    dcur = w.D[ps.cur, c + 1]
    arr = ps.t + dcur / w.v
    load = np.minimum(obs[c], ps.cap)
    F = np.zeros((len(c), N_FEAT), np.float32)
    F[:, 0] = dcur / w.Dmax
    F[:, 1] = w.D[c + 1, 0] / w.Dmax
    F[:, 2] = (w.due[c] - arr) / w.shift
    F[:, 3] = ps.cap / w.Q
    F[:, 4] = (w.shift - ps.t) / w.shift
    F[:, 5] = (~ps.visited).mean()
    if len(c) > 1:
        sub = w.D[np.ix_(c + 1, c + 1)]
        np.fill_diagonal(sub, np.inf)
        kk = min(3, len(c) - 1)
        F[:, 6] = np.sort(sub, 1)[:, :kk].mean(1) / w.Dmax
    carrier = day["carriers"][ps.k]
    base = w.freight(carrier, ps.W)
    if joint:
        F[:, 7] = np.clip(obs[c] / w.Q, 0, 2)
        F[:, 8] = load / w.Q
        F[:, 9] = [(w.freight(carrier, ps.W + l * uw) - base) for l, uw in zip(load, w.unit_w[c])]
        F[:, 9] = (F[:, 9] + w.p["c_km"] * dcur) / 10.0
    else:
        F[:, 9] = w.p["c_km"] * dcur / 10.0
    return F


class MLP:
    """Tiny numpy MLP (F -> H -> H -> 1, ReLU) with Adam; parameter count independent of n."""

    def __init__(self, rng, sizes=(N_FEAT, 32, 32, 1), lr=1e-3):
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.lr = lr
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def n_params(self):
        return int(sum(p.size for p in self.W + self.b))

    def copy_from(self, o):
        self.W = [x.copy() for x in o.W]; self.b = [x.copy() for x in o.b]

    def forward(self, X):
        h = X
        self.cache = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.maximum(z, 0) if i < len(self.W) - 1 else z
            self.cache.append(h)
        return h[:, 0]

    def backward(self, g):
        g = g[:, None].astype(np.float32)
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            a = self.cache[i]
            gW[i] = a.T @ g
            gb[i] = g.sum(0)
            if i > 0:
                g = (g @ self.W[i].T) * (self.cache[i] > 0)
        self.t += 1
        b1, b2 = 0.9, 0.999
        params = self.W + self.b
        grads = gW + gb
        for k, (p, gr) in enumerate(zip(params, grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * gr
            self.v[k] = b2 * self.v[k] + (1 - b2) * gr * gr
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + 1e-8)


class NeuralQ:
    """DQN with a shared candidate-scoring MLP: Q(s,a) = MLP(features(s,a)). Experience replay,
    target network, Huber loss, linearly decaying epsilon."""

    def __init__(self, joint, rng, n_max, gamma=0.9, buffer=20000, batch=64, warmup=500,
                 target_every=500, eps_start=1.0, eps_end=0.05, updates_per_step=1, hidden=32, lr=1e-3,
                 rw=None, decay_frac=0.6):
        self.rw = rw or REWARD
        self.decay_frac = decay_frac
        self.td_log = []                        # revision: training diagnostics
        self.joint, self.rng, self.gamma = joint, rng, gamma
        self.net, self.tgt = MLP(rng, sizes=(N_FEAT, hidden, hidden, 1), lr=lr), MLP(rng, sizes=(N_FEAT, hidden, hidden, 1), lr=lr)
        self.tgt.copy_from(self.net)
        self.buf_x = np.zeros((buffer, N_FEAT), np.float32)
        self.buf_r = np.zeros(buffer, np.float32)
        self.buf_nx = np.zeros((buffer, n_max, N_FEAT), np.float32)
        self.buf_nm = np.zeros((buffer, n_max), bool)
        self.buf_done = np.zeros(buffer, bool)
        self.size_, self.ptr, self.cap_ = 0, 0, buffer
        self.batch, self.warmup, self.target_every = batch, warmup, target_every
        self.eps_start, self.eps_end, self.eps = eps_start, eps_end, eps_start
        self.updates, self.upd_per_step = 0, updates_per_step
        self.training = True
        self.name = "neural_" + ("joint" if joint else "separate")

    def choose(self, ps, c):
        F = cand_features(ps, c, self.joint)
        if self.training and self.rng.random() < self.eps:
            i = int(self.rng.integers(len(c)))
        else:
            q = self.net.forward(F)
            i = int(np.argmax(q))
        return int(c[i]), (F, i)

    def plan(self, world, day):
        trips, self._traces = plan_day(world, day, self.choose)
        return trips

    def learn(self, world, outcome, traces=None):
        traces = traces or self._traces
        rs = rewards(world, outcome, self.joint, getattr(self, "rw", REWARD))
        n_steps = 0
        for trace, r in zip(traces, rs):
            for i, (F, a) in enumerate(trace):
                p = self.ptr
                self.buf_x[p] = F[a]
                self.buf_r[p] = r[i]
                self.buf_nm[p] = False
                if i + 1 < len(trace):
                    F2 = trace[i + 1][0]
                    self.buf_nx[p, :len(F2)] = F2
                    self.buf_nm[p, :len(F2)] = True
                    self.buf_done[p] = False
                else:
                    self.buf_done[p] = True
                self.ptr = (p + 1) % self.cap_
                self.size_ = min(self.size_ + 1, self.cap_)
                n_steps += 1
        if self.size_ >= self.warmup:
            for _ in range(n_steps * self.upd_per_step):
                self._update()

    def _update(self):
        idx = self.rng.integers(0, self.size_, self.batch)
        x, r, done = self.buf_x[idx], self.buf_r[idx], self.buf_done[idx]
        nx, nm = self.buf_nx[idx], self.buf_nm[idx]
        B, N, Fd = nx.shape
        qn = self.tgt.forward(nx.reshape(-1, Fd)).reshape(B, N)
        qn = np.where(nm, qn, -np.inf).max(1)
        qn = np.where(done | ~np.isfinite(qn), 0.0, qn)
        y = r + self.gamma * qn
        q = self.net.forward(x)
        d = q - y
        g = np.clip(d, -1.0, 1.0) / self.batch      # Huber gradient
        ad = np.abs(d)
        self.td_log.append((float(np.where(ad <= 1, 0.5 * d * d, ad - 0.5).mean()), float(q.mean())))
        self.net.backward(g)
        self.updates += 1
        if self.updates % self.target_every == 0:
            self.tgt.copy_from(self.net)

    def set_progress(self, frac, decay_frac=None):
        decay_frac = self.decay_frac if decay_frac is None else decay_frac
        self.eps = max(self.eps_end, self.eps_start - (self.eps_start - self.eps_end) * frac / decay_frac)
