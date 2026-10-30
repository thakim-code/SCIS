# Decision-Centric IoT + Blockchain SCIS — reproducibility package

Generates every number, table and figure in Sections 5–6 of the manuscript
"A Decision-Centric IoT and Blockchain-Enabled Supply Chain Information System for
Demand-Aware Routing: Architecture and Multi-Seed Evaluation".

## Data
Supply Chain Logistics Problem Dataset (Kalganova & Dzalbs, Brunel University London, 2019),
CC BY 4.0, https://doi.org/10.17633/rd.brunel.7558679.v2 . Download the Excel file and pass its
path with `--data`. The dataset is not redistributed here.

## Install
Python 3.10+ with numpy, pandas, openpyxl, scipy, matplotlib (`pip install -r requirements.txt`).
No GPU and no deep-learning framework are needed: the neural router is a small numpy MLP.

## Run everything (about 15 minutes on one CPU core)
    bash run_all.sh "Supply_chain_logisitcs_problem.xlsx"

or step by step:

    python run_main.py    --data DATA.xlsx --seeds 0-29   # Tables 4-7, sensitivity, Figs 6-11
    python run_ledger.py  --data DATA.xlsx                # Table 8 (tamper + invalid-decision tests)
    python run_scaling.py --data DATA.xlsx                # Table 9, Fig 12
    python analyze.py                                     # results/summary.json, results/tables.md, figures/

Every run writes one JSON per seed (atomically). Re-running the same command skips finished seeds,
so an interrupted run (e.g. a power cut) resumes where it stopped.

## Layout
    scs/data.py      loads the dataset; derives demand pools, unit weights, service levels,
                     carrier shares and weight-banded freight rates (all real)
    scs/env.py       world/layout, day generation, IoT noise, trip planning state, execution
                     against true demand, metrics, 2-opt
    scs/policies.py  reactive nearest-neighbour, NN + 2-opt, demand-aware + 2-opt,
                     tabular Q-learning, neural candidate-scoring DQN (numpy)
    scs/ledger.py    SHA-256 hash-chained ledger, 5 validators, majority vote, trust EMA
    run_main.py      main 30-seed experiment + sensitivity (seeds 0-11)
    run_ledger.py    tamper detection and invalid-decision injection
    run_scaling.py   10-160 customers
    analyze.py       statistics (paired Wilcoxon, Holm), tables, figures
    make_figure3.py  draws Figure 3 (framework diagram) with an automatic text-overlap check

## Seeds
World layout 10000+s, test days 20000+s, policy init 30000+100s+k, training days 40000+s,
sensitivity 50000+s, ledger 60000-80000+s, scaling 90000-94000. Seeds s = 0..29.

## Assumed (not in the dataset) — see manuscript Section 5
Customer coordinates (uniform, 100 x 100 km, depot at centre), 3 vehicles/day, per-trip capacity =
0.55 x mean total daily demand / 3, 40 km/h, 0.25 h service, 8 h shift, due times DTD 3 h /
DTP 5 h / CRF 8 h, local cost 0.02 per km, IoT noise 5 % with 2 % missing readings.
All are in `scs/env.py::P`.

Licence: MIT.

## First-round revision additions (all reproducible; same seeds as above)
    pip install ortools cryptography            # OR-Tools baseline; Ed25519 timings for the BFT study
    python rev_main.py     --data DATA.xlsx --seeds 0-29 --workers 2   # ALNS, OR-Tools, neural+2-opt, coverage,
                                                                     # emissions proxy, learning diagnostics,
                                                                     # router-fault / validate-and-repair tests
    python rev_variants.py --data DATA.xlsx --group constraints,sens,coverage,hparams --workers 2
    python rev_variants.py --data DATA.xlsx --group frontier --workers 1   # timing-sensitive: run alone
    python rev_ledger.py   --data DATA.xlsx                             # PBFT scaling + evidence-based trust
    python rev_rollout.py  --data DATA.xlsx --seeds 0-29 --workers 2    # learned engine + decision-time rollout,
                                                                     # ALNS seeded by the learned rollout plan
    python rev_rollout_frontier.py --data DATA.xlsx                     # rollout on the latency plane (run alone)
    python rev_robust.py   --data DATA.xlsx --workers 2                 # rollout and seeded ALNS under the
                                                                     # sens and constraints scenarios
    python rev_robust_analyze.py                                        # results/rev_robust_summary.json
    python rev_cov_refs.py --data DATA.xlsx                             # dispatch reference for the coverage frontier
    python rev_analyze.py                                               # results/rev_summary.json, figures/rev_*.png

    scs/revision.py  ALNS (exact objective), OR-Tools GLS (prize-collecting CVRP), post-2-opt wrapper,
                     coverage/fairness metrics; env.py gains optional hard time windows, heterogeneous
                     capacities, clustered layouts, due-time scaling and demand shift (defaults unchanged,
                     so every number of the original submission is reproduced exactly).
    RolloutPolicy    (scs/revision.py) decision-time rollout of the learned router: top-K candidates completed by
                     the router and scored with the exact daily objective; no retraining.
    rev_ledger.py    PBFT discrete-event simulation with measured Ed25519/SHA-256/validation costs;
                     evidence-based (verifiable-witness) validation and trust.
ALNS uses a fixed iteration budget (150 per day in the main experiment), so its plans are reproducible;
the latency-frontier experiment uses wall-clock budgets and depends on the machine.

## Second-round (minor) revision additions
    python rev_ortools.py  --data DATA.xlsx --seeds 10-29 --workers 2   # OR-Tools GLS on the remaining seeds
    python rev_analyze.py                                               # adds "vs_ortools" to results/rev_summary.json

`rev_main.py` now runs OR-Tools on every seed. The first-round result files held OR-Tools for seeds 0-9 only;
`rev_ortools.py` adds it to the existing files for seeds 10-29 on the identical worlds and test days, with
the same evaluator as every other engine (execution against true demand, real weight-banded freight), so
no other result changes. `results/rev_summary.json` -> `vs_ortools` gives each engine's paired cost-per-unit
difference from OR-Tools over all 30 seeds (Table 8 and Section 4.1). OR-Tools stops on a 1 s wall-clock
limit, so its plans vary slightly with machine speed (seed 0: 0.2818 in the stored file, 0.2827 on a
re-run on a different machine).

## Result files

`results.zip` holds every per-seed result file and the summary JSON files, packed into one archive so that the
repository can be uploaded through the GitHub web interface. Unzip it in this folder before running
`rev_analyze.py` or `rev_robust_analyze.py`:

    unzip results.zip

## Figure files (named by manuscript figure number)
    figures/Figure3.png     make_figure3.py   framework diagram
    figures/Figure6.png     analyze.py        performance of all configurations
    figures/Figure7.png     analyze.py        relative difference from each baseline
    figures/Figure8.png     analyze.py        separate versus integrated reward
    figures/Figure9.png     analyze.py        training curves
    figures/Figure10.png    rev_analyze.py    learning diagnostics (shaded band: mean ± SD over seeds)
    figures/Figure11.png    rev_analyze.py    cost-coverage frontier
    figures/Figure12.png    analyze.py        sensitivity to IoT noise
    figures/Figure13.png    analyze.py        representation size and cost versus network size
    figures/Figure14.png    rev_analyze.py    decision quality versus decision latency
    figures/Figure15.png    rev_analyze.py    measured ledger time (values of Table 14)
    figures/FigureS1.png    rev_analyze.py    PBFT validation versus validator count (Supplementary Materials)
Figures 1, 2, 4 and 5 are conceptual diagrams and are not generated by code.

Production-proof changes (figures only; no result changed): the legends of Figures 7, 10, 12 and 15 were moved
outside the plotting area so that they no longer hide data, and Figure 15 is drawn from the same revision-round
timings as Table 14. The earlier first-round ledger figure (fig10_blockchain.png) was removed.
