# vfp-surrogate

A CLI tool for building and evaluating VFP (Vertical Flow Performance) surrogate models from reservoir simulation results.

## Overview

`vfp-surrogate` takes simulation output files and constructs machine learning surrogate models that approximate VFP tables. It supports multiple model types — from simple linear regression to symbolic regression via genetic programming — and can export results back to VFP format for use in reservoir simulators.

## Installation

```bash
pip install -e .
```

## Usage

The tool has two modes: `pipeline` and `evaluator`.

```
vfp-surrogate <mode> [options]
```

---

## Modes

### `pipeline` — Build a surrogate model

Reads simulation results, trains a surrogate model, and exports VFP tables.

```bash
vfp-surrogate pipeline \
  --input-file results.UNSMRY \
  --vfp-details-file details.json \
  --model linear \
  --output-folder ./output
```

#### Required arguments

| Argument | Description |
|---|---|
| `--input-file` | Path to simulation results file (`*.UNSMRY`) |
| `--vfp-details-file` | Path to VFP details file (`*.json`) |
| `--model` | Model type (see [Models](#models)) |
| `--output-folder` | Path to output directory |

#### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--well-data-filter-file` | — | Path to well data filter file (`*.json`) |
| `--table-granularity` | `5` | VFP records n-size |
| `--optimize-hyperparameters` | `False` | Enable hyperparameter tuning via Optuna |
| `--tuning-metric` | `mean_squared_error` | Metric to optimize during tuning |
| `--seed` | `None` | Random seed for reproducibility |

---

### `evaluator` — Evaluate an existing model

Evaluates VFP surrogate model performance against simulation results.

```bash
vfp-surrogate evaluator \
  --input-file results.UNSMRY \
  --output-folder ./output
```

#### Required arguments

| Argument | Description |
|---|---|
| `--input-file` | Path to simulation results file (`*.UNSMRY`) |
| `--output-folder` | Path to output directory |

#### Optional arguments

| Argument | Description |
|---|---|
| `--well-data-filter-file` | Path to well data filter file (`*.json`) |
| `--seed` | Random seed |

---

## Models

| Model | Key | Description |
|---|---|---|
| Linear | `linear` | Ordinary linear regression |
| Elastic Net | `elasticnet` | Regularized linear model (L1 + L2) |
| XGBoost | `xgb` | Gradient boosted trees |
| Gaussian Process | `gp` | Probabilistic surrogate with uncertainty estimates |
| Symbolic | `symbolic` | Genetic programming — evolves an interpretable mathematical expression |

### Symbolic model options

The symbolic model uses an island-model genetic algorithm and accepts additional tuning parameters:

| Argument | Default | Description |
|---|---|---|
| `--ga-generations` | `80` | Number of GA generations |
| `--ga-population` | `100` | Population size per island |
| `--n-islands` | `4` | Number of parallel islands |
| `--migration-interval` | `5` | Generations between island migrations |
| `--migration-size` | `3` | Individuals exchanged per migration |
| `--simplify-interval` | `5` | Generations between SymPy simplification passes (`0` to disable) |
| `--parsimony-coefficient` | `0.001` | Penalty per tree node (controls expression complexity) |
| `--max-tree-height` | `6` | Maximum depth of GP expression trees |

---

## Examples

**Train an XGBoost model with hyperparameter tuning:**
```bash
vfp-surrogate pipeline \
  --input-file sim.UNSMRY \
  --vfp-details-file vfp.json \
  --model xgb \
  --output-folder ./output \
  --optimize-hyperparameters \
  --tuning-metric r2_score \
  --seed 42
```

**Train a symbolic model with custom GA settings:**
```bash
vfp-surrogate pipeline \
  --input-file sim.UNSMRY \
  --vfp-details-file vfp.json \
  --model symbolic \
  --output-folder ./output \
  --ga-generations 150 \
  --ga-population 200 \
  --n-islands 8 \
  --seed 42
```

**Evaluate against simulation data:**
```bash
vfp-surrogate evaluator \
  --input-file sim.UNSMRY \
  --output-folder ./eval-output \
  --seed 42
```

---

## Input File Formats

- **`.UNSMRY`** — Eclipse/OPM reservoir simulator summary file containing simulation results.
- **`.json` (VFP details)** — Specifies VFP table structure and configuration.
- **`.json` (well data filter)** — Optional filter to restrict which wells or time steps are included in training.