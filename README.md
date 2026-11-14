# Bridging Update Coordination and Flatness in Federated Learning

This repository provides the training and evaluation code for the four main
experiment tables, the Fashion-MNIST coordination-order comparison, and the
numerical diagnostics. Training shares a backbone and keeps classifier heads and
batch-normalization buffers private. The adapters implement the coordination or
probe components used in the paper.

## Environment and data

Use Python 3.10 and a PyTorch 2.1.2 / torchvision 0.16.2 installation appropriate
for your CUDA environment. Install the remaining dependencies with
`python -m pip install -r requirements.txt`. Training runs on one device per
process. The first use of a pinned author component fetches its source from the
author repository and verifies SHA-256. Set `FL_AUTHOR_CACHE` to change the
default cache at `~/.cache/fl_coordination/authors`.

```bash
python -m fl_coordination prepare fashion_mnist --data-root datasets/fashion-mnist
python -m fl_coordination prepare cifar100 --data-root datasets/cifar100
python -m fl_coordination prepare office_home --data-root datasets/office-home
python -m fl_coordination prepare retina --data-root datasets/retina
```

Existing data roots can be passed directly to training. For Office-Home, use the
four domain folders `Art`, `Clipart`, `Product`, and `Real_World` with class
subfolders. For Retina, use the binary archive released by the
[FedLD authors](https://github.com/Zeng-Shuang/FedLD), including the image
arrays in `train/` and `test/`, the CSVs, and `5_clients/split_2/`.
The data loaders construct their splits from the requested seed. Retina
groups patients and duplicate images before allocating held-out examples.
Pretrained ResNet-18 weights are obtained through torchvision.

| Dataset | Clients | Backbone | Data epochs | Communication rounds |
| --- | ---: | --- | ---: | ---: |
| Fashion-MNIST | 5 | ResNet-18 | 200 | 1,800 |
| CIFAR-100 | 20 | ResNet-20 | 300 | 1,500 |
| Office-Home | 4 | ResNet-18 | 100 | 3,900 |
| Retina | 5 | ResNet-18 | 100 | 100 |

## Train and summarize

List finalized recipes with `python -m fl_coordination list cifar100`. Each
dataset contains FedAvg, the five ordinary coordinators, and the twenty
combinations, all with seeds 0, 1, and 2. Each invocation runs one job.

```bash
python -m fl_coordination train cifar100 --data-root datasets/cifar100 \
  --coordinator fedfv --seed 2
python -m fl_coordination train cifar100 --data-root datasets/cifar100 \
  --coordinator fedfv --mechanism asam --seed 2
python -m fl_coordination summarize cifar100 --coordinator fedfv --mechanism asam
```

Coordinator identifiers are `fedavg`, `fedfv`, `fedmgda`, `fedmdfg`, `fedld`,
and `fedawa`. Mechanism identifiers are `baseline` (the default), `sam`, `asam`,
`gf`, and `gloss`. On Fashion-MNIST the historical `sam` table row uses ASAM
with separate coordination and `asam` uses the same perturbation with direct
coordination. On the other datasets `sam` uses standard SAM with separate
coordination. FedGloSS uses an additional local SAM probe on CIFAR-100,
Office-Home, and Retina.

Train the corresponding same-seed ordinary coordinator before a recipe with
the `weight_refinement` endpoint. These recipes evaluate raw and refined
weights from a single trajectory. Validation data control the selection. The Office-Home and Retina recipes specify which checkpoint to report.
Immutable run metadata record the split and configuration. Metrics and model
weights are written separately under `runs/results` and `runs/checkpoints`.
Reusing the same options resumes the same run. `summarize` requires three
completed seeds and reports seed means, sample standard deviations, and
paired gains in percentage points. Retina additionally reports macro AUROC over clients whose test data contain both classes, with client counts for each seed.

## Numerical diagnostics

Run the Fashion-MNIST seed-2 training recipes first. The terminal-state
checkpoint is `last.pt`. Give each diagnostic command its matching result
directory and checkpoint.

- `branch-replay` uses FedAWA-Agg with separately coordinated ASAM
  (`--mechanism sam`). It reconstructs the nine replay blocks and returns
  inner products, direction differences and per-client descent predictions.
- `sensitivity` uses FedAvg, FedAWA-Agg, and directly coordinated ASAM
  (`--mechanism asam`) in three separate invocations. It reports the macro
  validation loss increase after shared-backbone gradient ascent.
- The training logs provide validation accuracy, communication rounds, conflict
  statistics and measured training/validation time for the trajectory and
  coordination-cost analysis.

```bash
python -m fl_coordination branch-replay --data-root datasets/fashion-mnist \
  --run runs/results/fashion_mnist/seed_2/fedawa/sam \
  --checkpoint runs/checkpoints/fashion_mnist/seed_2/fedawa/sam/last.pt
```
