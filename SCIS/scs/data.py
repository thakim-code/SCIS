"""Load the Brunel Supply Chain Logistics Problem dataset (Kalganova & Dzalbs, 2019)
and derive every real quantity the simulation uses. Nothing here is synthetic."""
import numpy as np
import pandas as pd

DEPOT = "PLANT03"


def load(path):
    x = pd.ExcelFile(path)
    o = x.parse("OrderList")
    fr = x.parse("FreightRates")
    cap = x.parse("WhCapacities")
    p = o[o["Plant Code"] == DEPOT].copy()
    cap_col = [c for c in cap.columns if c.strip() == "Daily Capacity"][0]

    winsor = float(np.percentile(p["Unit quantity"], 95))
    custs = sorted(p["Customer"].unique())
    pools, unit_w, svc, n_orders = [], [], [], []
    for c in custs:
        q = p.loc[p.Customer == c, "Unit quantity"].to_numpy(float)
        w = p.loc[p.Customer == c, "Weight"].to_numpy(float)
        pools.append(np.minimum(q, winsor))
        ok = q > 0
        unit_w.append(float(np.median(w[ok] / q[ok])) if ok.any() else 0.0)
        svc.append(p.loc[p.Customer == c, "Service Level"].mode().iloc[0])
        n_orders.append(len(q))

    shares = p["Carrier"].value_counts(normalize=True)
    carriers = ["V444_0", "V444_1", "V44_3"]
    carrier_share = np.array([shares.get(c, 0.0) for c in carriers])
    carrier_share = carrier_share / carrier_share.sum()

    # Weight-banded rate tables, PORT04 -> PORT09 (the only lane PLANT03 uses).
    # V444_0 ships almost exclusively DTP, V444_1 exclusively DTD (OrderList cross-tab).
    # Where a bracket has several rate rows (different transit-day tiers), the cheapest is used.
    tables = {}
    for car, svc_cd in (("V444_0", "DTP"), ("V444_1", "DTD")):
        t = fr[(fr.Carrier == car) & (fr.orig_port_cd == "PORT04") & (fr.dest_port_cd == "PORT09")
               & (fr.svc_cd == svc_cd)]
        t = (t.groupby("minm_wgh_qty").agg(min_cost=("minimum cost", "min"), rate=("rate", "min"))
              .reset_index().sort_values("minm_wgh_qty"))
        tables[car] = (t["minm_wgh_qty"].to_numpy(float), t["min_cost"].to_numpy(float),
                       t["rate"].to_numpy(float))

    return dict(customers=custs, pools=pools, unit_w=np.array(unit_w), svc=np.array(svc),
                n_orders=np.array(n_orders), winsor=winsor, carriers=carriers,
                carrier_share=carrier_share, tables=tables,
                depot_daily_capacity=int(cap.loc[cap["Plant ID"] == DEPOT, cap_col].iloc[0]),
                n_orders_total=len(o), n_orders_depot=len(p), n_customers_total=o.Customer.nunique())


def freight(tables, carrier, weight):
    """Real weight-banded freight charge: max(minimum cost, rate x weight) in the applicable bracket.
    Carrier V44_3 is customer-referred freight (CRF): no charge to the shipper."""
    if weight <= 0 or carrier == "V44_3":
        return 0.0
    mins, mc, rt = tables[carrier]
    i = max(0, np.searchsorted(mins, weight, side="right") - 1)
    cost = max(mc[i], rt[i] * weight)
    if i + 1 < len(mins):  # weight in a gap/below next bracket: never pay more than shipping at the next bracket
        cost = min(cost, max(mc[i + 1], rt[i + 1] * mins[i + 1]))
    return float(cost)
