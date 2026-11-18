# GreenWheel-Opt

[![DOI](https://zenodo.org/badge/1249816989.svg)](https://doi.org/10.5281/zenodo.20396939)

A graph neural network (GNN) surrogate for multi-objective optimization of
green-energy wheeling allocation. The framework formulates the allocation of
renewable generation to corporate consumers, under sub-hourly temporal
matching, as a three-objective problem (retailer profit, surplus electricity,
RE100 shortfall) and accelerates population-based multi-objective evolutionary
algorithms (NSGA-II, MOEA/D, NSGA-III) with a heterogeneous GNN pre-screening
surrogate (WheelingGNN).

## Associated paper

This repository accompanies the paper:

> Ming-I Chen, Ying-Lin Hsu, Yi-Hsin Chen.
> "When Does a Graph Neural Network Surrogate Pay Off in Green Energy Wheeling Allocation?"

If you use this code or data, please cite the paper (see `CITATION.cff`).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Repository layout

- `src/greenwheel/`: problem formulation, simulator, optimizers, and surrogate models
- `scripts/`: data generation, surrogate training, experiments, and figures
- `configs/`: configuration files
- `checkpoints/`: trained WheelingGNN surrogate checkpoints (synthetic and semi-real)
- `data/surrogate/*/metadata.json`: dataset split metadata (seeds and partitions)
- `docs/case_study_protocol.md`: protocol of the 2021 backtest and its amendments

## Reproducing the results

Training datasets are large and are regenerated from seeds rather than stored:

```bash
# 1. Generate training data (synthetic)
python scripts/generate_training_data.py --config configs/surrogate.yaml

# 2. Train the GNN surrogate
python scripts/train_surrogate.py --data data/surrogate/phase7_synthetic \
    --config configs/surrogate.yaml --checkpoint-dir checkpoints/surrogate_phase7_synthetic

# 3. Run the main multi-algorithm comparison
python scripts/run_phase7_experiments.py --exp 7a \
    --model checkpoints/surrogate_phase7_synthetic/best.pt
```

Semi-real instances use public Taipower generation data; see
`scripts/download_taipower.py`. Random seeds and train/validation/test
partitions are recorded in the `metadata.json` files.

### Surrogate reliability and screening behaviour

Four further studies characterise when pre-screening helps and when it does not.
Each shares the seeds and the per-instance reference-point protocol of the main
semi-real experiment, so their numbers are directly comparable to it.

```bash
# Rank agreement between surrogate and simulator measured during the run,
# on the full-evaluation generations where exact labels are already available.
python scripts/run_sp_trajectory.py --scales medium,large --n-seeds 5

# Whether retaining screened-out offspring with predicted objectives, instead
# of discarding them, removes the harm caused by surrogate false negatives.
python scripts/run_recoverability_experiment.py --scales small,medium,large --n-seeds 10

# Optimization quality on semi-real instances at the largest scale tested
# (50 generators x 100 consumers, 5,000 decision variables). Shardable by seed.
python scripts/run_xlarge_semireal.py --device cuda --seeds 0,1

# Attention coefficients of the deployed model, and whether they concentrate
# on generator-consumer pairs that carry economically meaningful edge features.
python scripts/run_attention_analysis.py --scale medium --n-seeds 5
```

A reduced surrogate configuration (`configs/surrogate_tiny.yaml`) is provided for
the inference-cost analysis: it cuts the parameter count by roughly a factor of
six while matching the deployed model's held-out ranking accuracy.

### Equal-cost comparison, controlled degradation and baseline checks

```bash
# Anytime comparison of the six variants on the instances of the main comparison:
# hypervolume against exact evaluations and against wall-clock time, with the base
# algorithms run to 300 generations. Then the equal-cost summary and the figure.
python -u scripts/run_convergence.py \
    --model checkpoints/surrogate_phase7_synthetic/best.pt \
    --scales medium,large --n-seeds 10 --algos nsga2,moead,nsga3 \
    --vanilla-gens 300 --seed-base 70042 --ref table5 --device cpu \
    --output results/anytime_table5
python scripts/analyze_anytime.py --input results/anytime_table5
python scripts/generate_anytime_figure.py

# Controlled degradation of the surrogate's ranking at fixed instance, scale,
# algorithm and model: Gaussian noise on the predictions at six levels, with the
# rank agreement of the degraded predictions measured inside each run.
for scale in medium large; do for i in 0 1 2 3 4 5 6 7 8 9; do
  python -u scripts/run_noise_control.py --scale $scale --seed-index $i \
      --output results/noise_control
done; done
python scripts/run_noise_control.py --analyze --output results/noise_control

# Whether the DeepSets baseline delivers the GNN's gain inside SA-NSGA-II.
# Needs the outputs of the GNN-versus-MLP experiment and of the baseline training
# (the MLP checkpoint, its results file and the DeepSets metrics).
python scripts/train_deepsets_checkpoint.py
python -u scripts/run_deepsets_downstream.py --output results/deepsets_downstream

# Held-out accuracy of the flat baselines with their full input and with the
# allocation matrix plus instance-level features only (three MLP seeds).
python -u scripts/eval_flat_baselines.py --output results/flat_baselines

# Whether replicating generators with noise, as the largest semi-real instances
# do, changes the surrogate's local ranking.
python -u scripts/run_replicate_effect.py --n-seeds 10 --output results/replicate_effect
```

The results in the paper were produced with PyTorch Geometric 2.7.0 and pymoo 0.6.1.6.
In that pymoo version the niching step of NSGA-III draws random numbers that the run's
seed does not fix, so NSGA-III runs are reproducible in distribution but not run for run.

## Case study: Taiwanese green-electricity retailers (2021 backtest)

The case study replays the monthly allocation decision of three green-electricity
retailers, sized after real 2025 retailers (archetypes `S`, `M`, `L`), on 2021
weather and load under the 2026 wheeling rules. Each month the retailer fixes its
allocation shares from scenarios built only from data available two months
earlier, and is judged on the settlement of the month that actually happened.
The settlement follows Taipower's operating rules point 13
(`src/greenwheel/simulator/settlement.py`).

### Public data

All inputs are public. Place them under `data/raw/` as follows:

| Source | Used for | Path |
|---|---|---|
| T-REC certificate transaction records, direct supply and wheeling, 2025 (<https://www.trec.org.tw/certification_trade_situation/direct_supply>) | Retailer sites, technologies and buyer contract volumes | `data/raw/trec/direct_supply_2025.json` |
| Taipower monthly solar station statistics (data.gov.tw dataset 29938) | Monthly energy per solar station | `data/raw/taipower_monthly/d693003.csv` |
| Taipower monthly wind turbine statistics (data.gov.tw dataset 29961) | Monthly energy per wind station | `data/raw/taipower_monthly/d693004.csv` |
| National 10-minute solar and wind generation, 2017-2022 (Zenodo, doi:10.5281/zenodo.7537890, CC BY 4.0) | Intraday generation shape per technology | `data/raw/taipower_zenodo_7537890/powerRatio_10min_2017Jan_2022Jun.csv` |
| EWELD industrial and commercial load dataset (figshare, doi:10.6084/m9.figshare.21893808.v3, CC BY 4.0) | 15-minute buyer load shapes | `data/raw/eweld/EWELD/Electricity Consumption/` (unzipped archive) |

The T-REC file is the JSON table the direct-supply page loads for year 2025 (all
rows). `scripts/build_trec_archetypes.py` turns it into
`data/raw/trec/trec_archetypes_2025.tsv`, the per-retailer site and buyer table the
case study reads. Regulatory parameters are coded as constants with their sources
in the comments: Taipower operating rules for wheeling and direct supply, point 13
(settlement), the 2026 (115) transmission and distribution fees
(`src/greenwheel/data/replay2021.py`), and the Taipower high-voltage time-of-use
tariff calendars (`src/greenwheel/simulator/settlement.py`).

### Commands

```bash
# 1. Build the retailer table and check settlement, instances and data
python scripts/build_trec_archetypes.py
python scripts/check_settlement.py            # point-13 settlement vs a literal transcription
python scripts/check_consumer_contracts.py    # one contract per buyer vs the same transcription
python scripts/check_realcase.py              # Taipower and EWELD instances vs the raw files
python scripts/check_replay2021.py            # batched multi-scenario settlement vs per-scenario
python scripts/check_retailer2021.py          # retailer supply, demand, lead time and baseline

# 2. Is the retailer problem multi-objective? 3 archetypes x 3 supply ratios x
#    months 1 and 7 x seeds 1-3, then the per-cell judgement (results/retailer/e2f)
for arch in L M S; do for rho in 0.05 0.1 0.3; do for m in 1 7; do for s in 1 2 3; do
  python scripts/run_e2f_objectives.py --arch $arch --month $m --seed $s --rho $rho
done; done; done; done
python scripts/judge_e2f.py

# 3. Surrogate training data: 180 instances from 2020 only (2021 is the held-out replay)
for arch in L M S; do for rho in 0.05 0.1 0.3; do
  for s in 11 12; do for m in 1 3 4 6 7 8 9 10 11; do
    python scripts/gen_retailer_surrogate_data.py --arch $arch --rho $rho --seed $s \
        --year 2020 --month $m --out data/surrogate/retailer
  done; done
  for m in 1 7; do
    python scripts/gen_retailer_surrogate_data.py --arch $arch --rho $rho --seed 13 \
        --year 2020 --month $m --out data/surrogate/retailer
  done
done; done

# 4. Per-buyer fulfilment labels, then the split by instance (seed 13 is validation)
python scripts/add_buyer_labels.py data/surrogate/retailer data/surrogate/retailer_buyer \
    --files $(cd data/surrogate/retailer && ls ?_rho*_s*_2020??.pkl)
python scripts/build_retailer_split.py data/surrogate/retailer_buyer --val-seeds 13

# 5. Train: targets z-scored within each instance, worst fulfilment read out as a
#    soft minimum over buyers, with the per-buyer loss
python scripts/train_surrogate.py --data data/surrogate/retailer_buyer \
    --instance-relative-targets --batch-by-instance --min-readout 1 \
    --buyer-loss-weight 0.33 --checkpoint-dir checkpoints/retailer_buyer

# 6. Ranking check on unseen 2021 instances (must print PASS for each archetype)
for arch in L M S; do
  python scripts/diag_retailer_local_rank.py --ckpt checkpoints/retailer_buyer \
      --arch $arch --rho 0.1 --month 7 --seeds 1 2 3 4 5
done

# 7. Unit costs per exact evaluation, surrogate screen and low-fidelity screen,
#    single-threaded on an otherwise idle machine
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    python scripts/check_retailer_surrogate_cost.py --min-readout 1 \
    --out results/retailer/unit_costs.json

# 8. Backtest: every cell judged in step 2 x the nine 2021 replay months x seeds 1-3
#    x six methods. The objectives per cell are in results/retailer/e2f/judgement.json
#    (profit and worst fulfilment in every cell). Cost is counted, not timed, so runs
#    may execute in parallel; use one thread per run.
for arch in L M S; do for rho in 0.05 0.1 0.3; do
  for m in 1 3 4 6 7 8 9 10 11; do for s in 1 2 3; do
    for meth in nsga2 sa_gnn sa_random sa_low ev oracle; do
      extra=""; [ $meth = oracle ] && extra="--budget-evals 50000"
      OMP_NUM_THREADS=1 python scripts/run_e4_backtest.py --arch $arch --rho $rho \
          --month $m --seed $s --method $meth --objs profit worst \
          --ckpt checkpoints/retailer_buyer --costs results/retailer/unit_costs.json $extra
    done
  done; done
done; done

# 9. Analysis: speed-up to target, hypervolume at budget fractions, realized NT$ effect
python scripts/analyze_e4_backtest.py --dir results/retailer/e4
```

The protocol of the backtest, as fixed before it was run, and the two amendments made
before any reported backtest are recorded with their dates in
`docs/case_study_protocol.md`.

Buyer prices are the one assumed input of the case. To repeat part of the backtest
with every buyer's price redrawn from the same range:

```bash
for p in redraw1 redraw2; do for arch in S M L; do
  for m in 1 7 11; do for s in 1 2 3; do
    OMP_NUM_THREADS=1 python scripts/run_e4_backtest.py --arch $arch --rho 0.1 \
        --month $m --seed $s --method nsga2 --objs profit worst --price $p \
        --costs results/retailer/unit_costs.json --out results/price_redraw/$p
  done; done
done; done
python scripts/analyze_price_redraw.py
```

`scripts/run_e2_conflict.py` is the earlier two-objective conflict test on single
months (Taipower generation with EWELD load, `src/greenwheel/data/realcase.py`); its
repair operator and random allocations are shared by the scripts above.
`replay2021.VeaCustomers` is not used by the case study.

## License

MIT License (see `LICENSE`).
