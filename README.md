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
> "A Graph Neural Network Surrogate for Multi-Objective Green Energy Wheeling Allocation."

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

## License

MIT License (see `LICENSE`).
