"""Additions for the first-round revision.

* ORToolsPolicy  - prize-collecting CVRP with route-duration limit and soft (or hard) due times,
                   solved per day by Google OR-Tools with guided local search, on the SAME scalarised
                   objective as the learned routers' reward (freight linearised per carrier).
* Post2Opt       - wraps any policy and applies 2-opt to each of its trips.
* constrained feasibility (hard time windows, heterogeneous capacities) via World params.
* coverage-bonus reward, long-run coverage / fairness metrics, emissions proxy (vehicle-km).
* training diagnostics (TD loss, mean Q, greedy validation checkpoints) without touching RNG streams.
"""
import time
import numpy as np
from .env import PlanState, two_opt, tour_len
from .policies import plan_day, REWARD

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
except Exception:  # pragma: no cover
    pywrapcp = None


# ------------------------------------------------------------------ OR-Tools baseline
class ORToolsPolicy:
    """Per-day prize-collecting capacitated VRP (OR-Tools, guided local search).

    min  sum_k a1*(c_km*dist_k + fbar_k*weight_k) + a2*sum_j lateness_j - sum_{j served} (a3*load_j/Q + a4)
    s.t. capacity, route duration <= shift (incl. return), each customer at most once,
         [hard_tw] arrival_j <= due_j.
    fbar_k is carrier k's real freight charge per unit weight at a full-load trip (linearisation).
    A post-fill step tops up each trip with a partial load of the best remaining customer, as the
    construction policies do, so the baseline is not penalised for unsplittable demand."""

    def __init__(self, time_limit=1.0, rw=REWARD, a_cov=0.0, hard_tw=False, scale=10000, name=None,
                 warm=None, first="PATH_CHEAPEST_ARC"):
        self.tl, self.rw, self.a_cov, self.hard_tw, self.S = time_limit, rw, a_cov, hard_tw, scale
        self.warm, self.first = warm, first     # warm: policy whose plan seeds the search
        self.name = name or "ortools"
        self.last_obj = None

    def plan(self, world, day):
        w, obs = world, day["obs"]
        cand = np.where(obs > 0)[0]
        nodes = [0] + [int(j) + 1 for j in cand]          # routing index -> world node
        N = len(nodes)
        K = w.K
        caps = [int(q) for q in w.Qk]
        dem = [0] + [int(round(min(obs[j], max(caps)))) for j in cand]
        uw = np.r_[0.0, w.unit_w[cand]]
        Wfull = w.Q * float(np.median(w.unit_w))
        fbar = [w.freight(day["carriers"][k], Wfull) / Wfull for k in range(K)]
        a1, a2, a3 = self.rw["a_cost"], self.rw["a_delay"], self.rw["a_dem"]
        S = self.S
        mgr = pywrapcp.RoutingIndexManager(N, K, 0)
        R = pywrapcp.RoutingModel(mgr)
        Dn = w.D[np.ix_(nodes, nodes)]
        cbs = []
        for k in range(K):
            arc = (S * a1 * (w.p["c_km"] * Dn + fbar[k] * (uw * np.array(dem))[None, :])).round().astype(int)
            cb = R.RegisterTransitMatrix(arc.tolist())
            cbs.append(cb)
            R.SetArcCostEvaluatorOfVehicle(cb, k)
        dcb = R.RegisterUnaryTransitVector(dem)
        R.AddDimensionWithVehicleCapacity(dcb, 0, caps, True, "cap")
        svc = np.r_[0.0, np.full(N - 1, w.svc_h)]
        tmat = (60 * (Dn / w.v + svc[:, None])).round().astype(int)   # minutes, service at origin
        tcb = R.RegisterTransitMatrix(tmat.tolist())
        H = int(60 * w.shift)
        R.AddDimension(tcb, 0, H, True, "time")
        T = R.GetDimensionOrDie("time")
        for r, j in enumerate(cand, start=1):
            idx = mgr.NodeToIndex(r)
            due = int(60 * w.due[j])
            if self.hard_tw:
                T.CumulVar(idx).SetMax(due)
            else:
                T.SetCumulVarSoftUpperBound(idx, due, max(1, int(round(S * a2 / 60))))
            R.AddDisjunction([idx], int(round(S * (a3 * dem[r] / w.Q + self.a_cov))))
        prm = pywrapcp.DefaultRoutingSearchParameters()
        prm.first_solution_strategy = getattr(routing_enums_pb2.FirstSolutionStrategy, self.first)
        prm.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        prm.time_limit.FromMilliseconds(int(1000 * self.tl))
        self.warm_obj = None
        if self.warm is not None:
            pos = {int(j): r for r, j in enumerate(cand, start=1)}
            init = []
            for k, t in enumerate(self.warm.plan(world, day)):
                r, load, tm, cur = [], 0, 0, 0       # trim to a route feasible in this model
                for j in t:
                    if j not in pos:
                        continue
                    q = dem[pos[j]]
                    tt = tm + tmat[cur, pos[j]]
                    back = tt + tmat[pos[j], 0]
                    if load + q > caps[k] or back > H or (self.hard_tw and tt > int(60 * w.due[j])):
                        break
                    r.append(pos[j]); load += q; tm = tt; cur = pos[j]
                init.append(r)
            R.CloseModelWithParameters(prm)
            a0 = R.ReadAssignmentFromRoutes(init, True)
            if a0 is not None:
                self.warm_obj = a0.ObjectiveValue()
                sol = R.SolveFromAssignmentWithParameters(a0, prm)
            else:
                sol = R.SolveWithParameters(prm)
        else:
            sol = R.SolveWithParameters(prm)
        self.last_obj = sol.ObjectiveValue() if sol is not None else None
        trips = [[] for _ in range(K)]
        if sol is not None:
            for k in range(K):
                i = R.Start(k)
                i = sol.Value(R.NextVar(i))
                while not R.IsEnd(i):
                    trips[k].append(int(nodes[mgr.IndexToNode(i)] - 1))
                    i = sol.Value(R.NextVar(i))
        return fill_trips(world, day, trips)


def fill_trips(world, day, trips):
    """Append, to each trip, the feasible unvisited customer with the largest deliverable (partial)
    load per km while capacity remains (same rule as the demand-aware construction)."""
    visited = np.zeros(world.n, bool)
    for t in trips:
        visited[t] = True
    out = []
    for k, seq in enumerate(trips):
        ps = PlanState(world, day, k, visited)
        for j in seq:  # replay to get time/capacity state (visited already set)
            arr = ps.t + world.D[ps.cur, j + 1] / world.v
            load = min(day["obs"][j], ps.cap)
            ps.seq.append(j); ps.loads.append(load); ps.arr.append(arr)
            ps.cap -= load; ps.W += load * world.unit_w[j]; ps.t = arr + world.svc_h; ps.cur = j + 1
        while True:
            c = ps.feasible()
            if len(c) == 0:
                break
            load = np.minimum(day["obs"][c], ps.cap)
            j = int(c[np.argmax(load / (world.D[ps.cur, c + 1] + 1.0))])
            ps.step(j)
        out.append(ps.seq)
    return out


class Post2Opt:
    """Apply 2-opt to each trip of a base policy (time-window aware when windows are hard)."""

    def __init__(self, base, name=None):
        self.base = base
        self.name = name or (getattr(base, "name", "policy") + "+2opt")

    def plan(self, world, day):
        return [tw_two_opt(world, t) for t in self.base.plan(world, day)]


def tw_two_opt(world, seq):
    s = two_opt(world, seq)
    if world.p.get("hard_tw") and not route_ok(world, s):
        return list(seq)
    return s


def route_ok(world, seq):
    t, cur = 0.0, 0
    for j in seq:
        t += world.D[cur, j + 1] / world.v
        if world.p.get("hard_tw") and t > world.due[j] + 1e-9:
            return False
        t += world.svc_h
        cur = j + 1
    return t + world.D[cur, 0] / world.v <= world.shift + 1e-9


# ------------------------------------------------------------------ metrics
def coverage_metrics(n, served_days):
    """served_days: list (one per day) of arrays of customers served that day."""
    T = len(served_days)
    cnt = np.zeros(n)
    last = np.full(n, -1)
    maxgap = np.zeros(n)
    for t, s in enumerate(served_days):
        for j in s:
            gap = t - last[j]
            maxgap[j] = max(maxgap[j], gap)
            last[j] = t
            cnt[j] += 1
    maxgap = np.maximum(maxgap, T - last)          # gap to end of horizon
    freq = cnt / T
    jain = freq.sum() ** 2 / (n * (freq ** 2).sum()) if freq.sum() > 0 else 0.0
    return dict(distinct_share=float((cnt > 0).mean()), jain=float(jain),
                max_gap_median=float(np.median(maxgap)), never_served=int((cnt == 0).sum()),
                p10_freq=float(np.percentile(freq, 10)))


def violations(world, obs, trips):
    """Count constraint violations that would be EXECUTED if a plan were not validated."""
    v = dict(capacity=0, duplicate=0, overtime=0, phantom=0)
    seen = set()
    for seq in trips:
        t, cur, load = 0.0, 0, 0.0
        for j in seq:
            if not (0 <= j < world.n):
                v["phantom"] += 1
                continue
            if j in seen:
                v["duplicate"] += 1
            seen.add(j)
            t += world.D[cur, j + 1] / world.v + world.svc_h
            cur = j + 1
        if t + world.D[cur, 0] / world.v > world.shift + 1e-9:
            v["overtime"] += 1
    return v


# ------------------------------------------------------------------ ALNS baseline
class ALNSPolicy:
    """Adaptive large neighbourhood search (Ropke & Pisinger, 2006) on the EXACT per-day objective
    used as the learned routers' reward, evaluated on observed demand:
        J = sum_k [a1*(c_km*dist_k + freight_k(weight_k)) + a2*sum lateness] - a3*delivered/Q - a4*stops
    with the real weight-banded freight, partial loads, route-duration (and optional hard due-time)
    feasibility. Destroy: random / worst / related (Shaw); repair: greedy best-insertion over a
    sampled candidate set, or regret-2. Simulated-annealing acceptance, adaptive operator weights,
    initialised from the demand-aware greedy + 2-opt plan; final 2-opt on every trip."""

    def __init__(self, time_limit=1.0, rw=REWARD, a_cov=0.0, seed=0, name=None, cand_sample=12,
                 init=None, max_iters=None):
        self.tl, self.rw, self.a_cov = time_limit, rw, a_cov
        self.init, self.max_iters = init, max_iters   # init: policy whose plan seeds the search
        self.rng = np.random.default_rng(seed)
        self.name = name or "alns"
        self.cs = cand_sample

    # route evaluation on observed demand
    def _route(self, w, day, k, seq):
        obs = day["obs"]
        cap, Wt, t, cur, dist, lat, dl = w.Qk[k], 0.0, 0.0, 0, 0.0, 0.0, 0.0
        hard = w.p.get("hard_tw")
        for j in seq:
            leg = w.D[cur, j + 1]
            dist += leg
            arr = t + leg / w.v
            if hard and arr > w.due[j] + 1e-9:
                return None
            l = obs[j] if obs[j] < cap else cap
            cap -= l; dl += l; Wt += l * w.unit_w[j]
            if arr > w.due[j]:
                lat += arr - w.due[j]
            t = arr + w.svc_h
            cur = j + 1
        dist += w.D[cur, 0]
        if t + w.D[cur, 0] / w.v > w.shift + 1e-9:
            return None
        r = self.rw
        cost = w.p["c_km"] * dist + w.freight(day["carriers"][k], Wt)
        return r["a_cost"] * cost + r["a_delay"] * lat - r["a_dem"] * dl / w.Q - self.a_cov * len(seq)

    def plan(self, world, day):
        from .policies import ratio
        w, rng = world, self.rng
        t_end = time.perf_counter() + self.tl
        if self.init is not None:
            cur = [list(s) for s in self.init.plan(w, day)]
        else:   # demand-aware greedy + 2-opt (2-opt move kept only if it respects hard windows)
            cur = [tw_two_opt(w, t) for t in plan_day(w, day, ratio)[0]]
        cv = [self._route(w, day, k, s) for k, s in enumerate(cur)]
        cv = [v if v is not None else 0.0 for v in cv]
        best, bv = [list(s) for s in cur], sum(cv)
        T0 = 0.05 * abs(bv) / max(1, len(sum(cur, []))) + 1e-6
        ops_d, ops_r = [self._d_random, self._d_worst, self._d_related], [self._r_greedy, self._r_regret]
        wd, wr = np.ones(3), np.ones(2)
        it = 0
        while time.perf_counter() < t_end and (self.max_iters is None or it < self.max_iters):
            it += 1
            i = rng.choice(3, p=wd / wd.sum()); j = rng.choice(2, p=wr / wr.sum())
            sol = [list(s) for s in cur]
            removed = ops_d[i](w, day, sol)
            ops_r[j](w, day, sol, removed)
            vals = [self._route(w, day, k, s) for k, s in enumerate(sol)]
            if any(v is None for v in vals):
                continue
            v = sum(vals)
            frac = it / self.max_iters if self.max_iters else (self.tl - (t_end - time.perf_counter())) / self.tl
            T = T0 * max(0.01, 1 - frac)
            score = 0.0
            if v < bv - 1e-9:
                best, bv, score = [list(s) for s in sol], v, 3.0
            if v < sum(cv) - 1e-9 or rng.random() < np.exp(-(v - sum(cv)) / T):
                cur, cv = sol, vals
                score = max(score, 1.0)
            wd[i] = 0.8 * wd[i] + 0.2 * score + 1e-3
            wr[j] = 0.8 * wr[j] + 0.2 * score + 1e-3
        self.iters = it
        out = []
        for k, s in enumerate(best):
            s2 = tw_two_opt(w, s)
            v1, v2 = self._route(w, day, k, s), self._route(w, day, k, s2)
            out.append(s2 if (v2 is not None and v1 is not None and v2 <= v1) else s)
        return out

    def _served(self, sol):
        return [(k, p) for k, s in enumerate(sol) for p in range(len(s))]

    def _nrem(self, sol):
        n = sum(len(s) for s in sol)
        return int(self.rng.integers(1, max(2, min(8, n) + 1))) if n else 0

    def _d_random(self, w, day, sol):
        q = self._nrem(sol)
        pos = self._served(sol)
        pick = self.rng.choice(len(pos), size=min(q, len(pos)), replace=False) if pos else []
        rem = sorted([pos[p] for p in pick], key=lambda x: -x[1])
        return [sol[k].pop(p) for k, p in rem]

    def _d_worst(self, w, day, sol):
        q = self._nrem(sol)
        gains = []
        for k, s in enumerate(sol):
            base = self._route(w, day, k, s) or 0.0
            for p in range(len(s)):
                v = self._route(w, day, k, s[:p] + s[p + 1:])
                if v is not None:
                    gains.append((base - v, k, p))
        gains.sort(reverse=True)
        rem = sorted([(k, p) for _, k, p in gains[:q]], key=lambda x: (x[0], -x[1]))
        out = []
        for k, p in sorted(rem, key=lambda x: -x[1]):
            out.append(sol[k].pop(p))
        return out

    def _d_related(self, w, day, sol):
        q = self._nrem(sol)
        allc = [j for s in sol for j in s]
        if not allc:
            return []
        seed = allc[int(self.rng.integers(len(allc)))]
        near = sorted(allc, key=lambda j: w.D[seed + 1, j + 1])[:q]
        for s in sol:
            s[:] = [j for j in s if j not in near]
        return near

    def _cands(self, w, day, sol, removed):
        used = set(j for s in sol for j in s)
        pool = [j for j in np.where(day["obs"] > 0)[0] if j not in used]
        extra = [j for j in pool if j not in removed]
        if len(extra) > self.cs:
            extra = list(self.rng.choice(extra, self.cs, replace=False))
        return list(dict.fromkeys(list(removed) + extra))

    def _best_ins(self, w, day, sol, j, vals):
        best = (np.inf, None, None)
        second = np.inf
        for k, s in enumerate(sol):
            for p in range(len(s) + 1):
                v = self._route(w, day, k, s[:p] + [j] + s[p:])
                if v is None:
                    continue
                d = v - vals[k]
                if d < best[0]:
                    second, best = best[0], (d, k, p)
                elif d < second:
                    second = d
        return best, second

    def _r_greedy(self, w, day, sol, removed):
        cands = self._cands(w, day, sol, removed)
        vals = [self._route(w, day, k, s) or 0.0 for k, s in enumerate(sol)]
        while cands:
            res = [(self._best_ins(w, day, sol, j, vals)[0], j) for j in cands]
            (d, k, p), j = min(res, key=lambda x: x[0][0])
            if k is None or d >= 0:
                break
            sol[k].insert(p, j); vals[k] += d; cands.remove(j)

    def _r_regret(self, w, day, sol, removed):
        cands = self._cands(w, day, sol, removed)
        vals = [self._route(w, day, k, s) or 0.0 for k, s in enumerate(sol)]
        while cands:
            res = []
            for j in cands:
                (d, k, p), sec = self._best_ins(w, day, sol, j, vals)
                if k is not None and d < 0:
                    res.append(((sec if np.isfinite(sec) else 1e9) - d, d, k, p, j))
            if not res:
                break
            _, d, k, p, j = max(res)
            sol[k].insert(p, j); vals[k] += d; cands.remove(j)


# ------------------------------------------------------------------ decision-time rollout of the learned router
class RolloutPolicy:
    """Learned router with decision-time rollout (policy improvement by simulation).

    At every stop the K highest-scoring admissible customers of the learned router are each tried; the rest of
    the day's plan is completed with the learned router itself, the completed plan is evaluated with the exact
    daily objective J_t on observed demand, and the candidate with the lowest J_t is committed. With K = 1 this is
    the learned router; by the rollout argument the result is no worse than the base policy on the plan objective."""

    def __init__(self, base, k=3, two_opt_final=True, name=None):
        self.base, self.k, self.t2 = base, k, two_opt_final
        self.name = name or f"rollout{k}"
        self._eval = ALNSPolicy(0.0)                # reuse exact route evaluation

    def _J(self, world, day, trips):
        v = 0.0
        for k, s in enumerate(trips):
            r = self._eval._route(world, day, k, s)
            if r is None:
                return np.inf
            v += r
        return v

    def _complete(self, world, day, prefix):
        """Plan the day with the base policy, forcing the given prefix (list of (trip, customer))."""
        it = iter(prefix)
        nxt = [next(it, None)]
        def choose(ps, c):
            p = nxt[0]
            if p is not None and p[0] == ps.k:
                nxt[0] = next(it, None)
                return p[1], None
            if p is not None and p[0] != ps.k:     # prefix continues on a later trip: end this one
                return None, None
            return self.base.choose(ps, c)[0], None
        return plan_day_forced(world, day, choose)

    def plan(self, world, day):
        from .policies import cand_features
        prefix = []
        while True:
            trips = self._complete(world, day, prefix)
            flat = [(k, j) for k, s in enumerate(trips) for j in s]
            if len(flat) <= len(prefix):
                break
            # state at the next decision: replay prefix to get the admissible set and scores
            k_next, _ = flat[len(prefix)]
            ps, c = state_after(world, day, prefix, k_next)
            if c is None or len(c) <= 1:
                prefix.append(flat[len(prefix)]); continue
            q = self.base.net.forward(cand_features(ps, c, self.base.joint))
            top = c[np.argsort(-q)[:self.k]]
            best, bv = None, np.inf
            for j in top:
                t = self._complete(world, day, prefix + [(k_next, int(j))])
                v = self._J(world, day, t)
                if v < bv:
                    best, bv = int(j), v
            prefix.append((k_next, best if best is not None else int(top[0])))
        trips = self._complete(world, day, prefix)
        if self.t2:
            out = []
            for k, s in enumerate(trips):
                s2 = tw_two_opt(world, s)
                a, b = self._eval._route(world, day, k, s), self._eval._route(world, day, k, s2)
                out.append(s2 if (a is not None and b is not None and b <= a) else s)
            trips = out
        return trips


def plan_day_forced(world, day, choose):
    """plan_day variant whose chooser may return None to close the current trip."""
    visited = np.zeros(world.n, bool)
    trips = []
    for k in range(world.K):
        ps = PlanState(world, day, k, visited)
        while True:
            c = ps.feasible()
            if len(c) == 0:
                break
            j, _ = choose(ps, c)
            if j is None:
                break
            if j not in set(c.tolist()):
                break
            ps.step(j)
        trips.append(ps.seq)
    return trips


def state_after(world, day, prefix, k_next):
    """PlanState of trip k_next after applying the prefix, and its admissible set."""
    visited = np.zeros(world.n, bool)
    for kk, j in prefix:
        if kk < k_next:
            visited[j] = True
    ps = PlanState(world, day, k_next, visited)
    for kk, j in prefix:
        if kk == k_next:
            ps.step(j)
    return ps, ps.feasible()
