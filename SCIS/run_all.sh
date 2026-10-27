#!/usr/bin/env bash
set -e
DATA="${1:?usage: bash run_all.sh path/to/Supply_chain_logisitcs_problem.xlsx}"
python run_main.py    --data "$DATA" --seeds 0-29
python run_ledger.py  --data "$DATA"
python run_scaling.py --data "$DATA"
python analyze.py
# first-round revision
python rev_main.py     --data "$DATA" --seeds 0-29 --workers 2
python rev_variants.py --data "$DATA" --group constraints,sens,coverage,hparams --workers 2
python rev_variants.py --data "$DATA" --group frontier --workers 1
python rev_ledger.py   --data "$DATA"
python rev_rollout.py  --data "$DATA" --seeds 0-29 --workers 2
python rev_rollout_frontier.py --data "$DATA"
python rev_robust.py   --data "$DATA" --workers 2
python rev_robust_analyze.py
python rev_cov_refs.py --data "$DATA"
python rev_analyze.py
