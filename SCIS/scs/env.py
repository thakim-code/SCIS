"""Simulation environment.

Real (from the dataset): customers, per-customer order-quantity distributions (winsorized at the
95th percentile), per-customer unit weights, service levels, carrier usage shares, and weight-banded
freight rates.  Assumed (disclosed): customer coordinates, fleet size, speed, service time, shift
length, due-time rule, local per-km cost, IoT noise model.
"""
import numpy as np
from .data import freight

P = dict(
    K=3,                 # vehicle trips per day
    cap_tightness=0.55,  # per-trip capacity = 0.55 x mean total daily demand / K
    area_km=100.0,       # customers uniform in an area x area square, depot at centre
    speed_kmh=40.0,
    service_h=0.25,
    shift_h=8.0,
    due_h={"DTD": 3.0, "DTP": 5.0, "CRF": 8.0},   # due time after dispatch by service level
    c_km=0.02,           # local delivery cost per km (dataset cost units)
    sigma_obs=0.05,      # IoT demand-observation noise (relative SD)
    p_missing=0.02,      # probability a customer's IoT reading is missing
    ewma=0.3,            # demand forecaster smoothing (used when a reading is missing)
)


class World:
    """Static instance for one seed: layout and customer set."""

    def __init__(self, data, rng, n_customers=None, params=None):
        self.p = dict(P, **(params or {}))
        n_all = len(data["customers"])
        if n_customers is None:
            idx = np.arange(n_all)
        else:  # scaling experiment: bootstrap customers from the real pool
            idx = rng.integers(0, n_all, n_customers)
        self.idx = idx
        self.n = len(idx)
        self.pools = [data["pools"][i] for i in idx]
        self.unit_w = data["unit_w"][idx]
        self.due = np.array([self.p["due_h"][s] for s in data["svc"][idx]]) * self.p.get("due_scale", 1.0)
        self.tables = data["tables"]
        self.carriers = data["carriers"]
        self.carrier_share = data["carrier_share"]
        a = self.p["area_km"]
        if self.p.get("layout", "uniform") == "clustered":   # revision: clustered layout
            nc = self.p.get("n_clusters", 4)
            ctr = rng.uniform(0.15 * a, 0.85 * a, (nc, 2))
            lab = rng.integers(0, nc, self.n)
            xy = np.clip(ctr[lab] + rng.normal(0, 0.08 * a, (self.n, 2)), 0, a)
        else:
            xy = rng.uniform(0, a, (self.n, 2))
        self.xy = np.vstack([[a / 2, a / 2], xy])            # node 0 = depot
        diff = self.xy[:, None, :] - self.xy[None, :, :]
        self.D = np.sqrt((diff ** 2).sum(-1))                  # km, (n+1)x(n+1)
        self.Dmax = self.D.max()
        self.mean_dem = np.array([pl.mean() for pl in self.pools])
        self.K = self.p["K"] if n_customers is None else max(1, int(round(self.p["K"] * self.n / n_all)))
        self.Q = self.p["cap_tightness"] * self.mean_dem.sum() / self.K
        # revision: heterogeneous fleet (multipliers average to 1, so total capacity is unchanged)
        mix = self.p.get("cap_mix")
        if mix:
            m = np.resize(np.array(mix, float), self.K)
            self.Qk = list(self.Q * m / m.mean())
        else:
            self.Qk = [self.Q] * self.K
        self.v = self.p["speed_kmh"]
        self.svc_h = self.p["service_h"]
        self.shift = self.p["shift_h"]

    def freight(self, carrier, w):
        return freight(self.tables, self.carriers[carrier], w)

    def draw_day(self, rng, sigma=None, p_missing=None, forecast=None):
        sigma = self.p["sigma_obs"] if sigma is None else sigma
        p_missing = self.p["p_missing"] if p_missing is None else p_missing
        true = np.array([pl[rng.integers(len(pl))] for pl in self.pools]) * self.p.get("dem_mult", 1.0)
        obs = true * np.maximum(0.0, 1.0 + rng.normal(0, sigma, self.n))
        miss = rng.random(self.n) < p_missing
        fc = self.mean_dem if forecast is None else forecast
        obs[miss] = fc[miss]
        carriers = rng.choice(len(self.carriers), self.K, p=self.carrier_share)
        return dict(true=true, obs=obs, miss=miss, carriers=carriers)


class PlanState:
    """State of one vehicle trip while it is being planned (uses observed demand only)."""

    def __init__(self, world, day, k, visited):
        self.w, self.day, self.k = world, day, k
        self.visited = visited          # shared across the day's trips
        self.cur, self.t, self.cap = 0, 0.0, world.Qk[k]
        self.seq, self.loads, self.arr = [], [], []
        self.W = 0.0                    # planned weight

    def feasible(self):
        w = self.w
        if self.cap <= 1e-9:
            return np.empty(0, int)
        c = np.where(~self.visited & (self.day["obs"] > 0))[0]
        if len(c) == 0:
            return c
        arr = self.t + w.D[self.cur, c + 1] / w.v
        ok = arr + w.svc_h + w.D[c + 1, 0] / w.v <= w.shift + 1e-9
        if w.p.get("hard_tw"):                       # revision: hard time windows
            ok &= arr <= w.due[c] + 1e-9
        return c[ok]

    def step(self, j):
        w = self.w
        arr = self.t + w.D[self.cur, j + 1] / w.v
        load = min(self.day["obs"][j], self.cap)
        self.seq.append(int(j)); self.loads.append(float(load)); self.arr.append(float(arr))
        self.cap -= load
        self.W += load * w.unit_w[j]
        self.t = arr + w.svc_h
        self.cur = j + 1
        self.visited[j] = True


def execute(world, day, trips):
    """Execute planned trips against TRUE demand. Returns per-trip, per-stop outcomes."""
    out = []
    for k, seq in enumerate(trips):
        cap, W, t, cur, dist = world.Qk[k], 0.0, 0.0, 0, 0.0
        stops = []
        prevF = 0.0
        for j in seq:
            leg = world.D[cur, j + 1]
            dist += leg
            arr = t + leg / world.v
            deliv = min(day["true"][j], cap)
            cap -= deliv
            W += deliv * world.unit_w[j]
            F = world.freight(day["carriers"][k], W)
            delay = max(0.0, arr - world.due[j])
            stops.append(dict(j=int(j), delivered=float(deliv), delay=float(delay),
                              dcost=float(world.p["c_km"] * leg + (F - prevF))))
            prevF = F
            t = arr + world.svc_h
            cur = j + 1
        ret = world.D[cur, 0] if seq else 0.0
        dist += ret
        if stops:
            stops[-1]["dcost"] += world.p["c_km"] * ret
        cost = world.p["c_km"] * dist + prevF
        out.append(dict(stops=stops, dist=float(dist), cost=float(cost), weight=float(W)))
    return out


def tour_len(world, seq):
    if not seq:
        return 0.0
    nodes = [0] + [j + 1 for j in seq] + [0]
    return float(sum(world.D[a, b] for a, b in zip(nodes[:-1], nodes[1:])))


def two_opt(world, seq):
    """Classical 2-opt on a single trip (depot-anchored)."""
    best = list(seq)
    if len(best) < 3:
        return best
    improved = True
    while improved:
        improved = False
        nodes = [0] + [j + 1 for j in best] + [0]
        n = len(nodes)
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                a, b, c, d = nodes[i - 1], nodes[i], nodes[k], nodes[k + 1]
                if world.D[a, c] + world.D[b, d] < world.D[a, b] + world.D[c, d] - 1e-9:
                    nodes[i:k + 1] = nodes[i:k + 1][::-1]
                    improved = True
        best = [x - 1 for x in nodes[1:-1]]
    return best


def day_metrics(world, day, trips, outcome):
    delivered = sum(s["delivered"] for tr in outcome for s in tr["stops"])
    total = float(day["true"].sum())
    delays = [s["delay"] for tr in outcome for s in tr["stops"]]
    cost = sum(tr["cost"] for tr in outcome)
    act = sum(tr["dist"] for tr in outcome)
    ref = sum(tour_len(world, two_opt(world, seq)) for seq in trips)
    return dict(cost=cost, delivered=delivered, ds=delivered / total if total else 0.0,
                delay=float(np.mean(delays)) if delays else 0.0,
                cpu=1000.0 * cost / delivered if delivered else np.nan,
                re=ref / act if act else 1.0, stops=len(delays), km=act,
                km_pu=1000.0 * act / delivered if delivered else np.nan)
