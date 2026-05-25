# Point-JR

**Pattern Clustering for Layout Hotspot Detection via Point Cloud and Graph Representations**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/pytorch-2.5-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Abstract

This repository provides the official implementation of **Point-JR**, a framework for layout pattern clustering and hotspot discrimination in EDA/VLSI design. Point-JR leverages point cloud and graph representations of IC layout patterns, combined with self-supervised and contrastive learning, to achieve effective pattern clustering without requiring labeled data for pre-training.

## Method Overview

Point-JR supports multiple data representations and model architectures:

| Model | Data Type | Description |
|:------|:----------|:------------|
| **SimSiam** | Image (256×256) | Self-supervised learning with ResNet-18 backbone |
| **LPA** | Image (1200×1200) | Triplet-based contrastive learning with custom CNN |
| **PointNeXt** | Point Cloud (4D) | Self-supervised learning on (x, y, n_x, n_y) point clouds |
| **Hybrid Point** | Point Cloud (4D) | SimSiam SSL + classification on PointNeXt features |
| **Hybrid Image** | Image (256×256) | SimSiam SSL + classification on ResNet features |
| **GNN** | Graph | Supervised learning with location-aware message passing |

## Repository Structure

```
point-jr/
├── train.py                          # Training entry point
├── inference.py                      # Feature extraction entry point
├── config.yaml                       # Training / inference configuration
├── environment.yml                   # Conda environment specification
│
├── models/                           # Model implementations
│   ├── simsiam.py                    #   SimSiam (image-based)
│   ├── point_simsiam.py              #   SimSiam (point cloud-based)
│   ├── pointnext.py                  #   PointNeXt encoder (PyG-based)
│   ├── hybrid.py                     #   Hybrid SSL + classification head
│   ├── lpa.py                        #   LPA feature extractor
│   └── gnn.py                        #   DATE-style GNN with edge updates
│
├── dataset/                          # Dataset loaders & augmentation
│   ├── dataset.py                    #   Point / Image / Graph / LPA datasets
│   └── augmentation.py               #   Point cloud & image augmentations
│
├── utils/                            # Training utilities
│   ├── utils.py                      #   Config, model builder, checkpointing
│   ├── criterion.py                  #   Loss functions
│   └── cluster.py                    #   Adaptive greedy clustering
│
├── scripts/
│   └── preprocess/
│       ├── gds2point.py              #   GDS → 4D point cloud conversion
│       └── gds2graph.py              #   GDS → PyG graph conversion
│
└── data/
    └── ICCAD2012/
        ├── raw/                      #   Original GDS files (included)
        │   ├── 1/ ~ 5/               #     chip directories
        │   │   ├── train.gds
        │   │   └── test.gds
        └── splits/                   #   Train / val / test CSV splits (included)
            ├── default/
            ├── mini_balance/
            └── mini_imbalance/
```

## Getting Started

### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate point-jr
```

**Key dependencies:** PyTorch 2.5.1, PyTorch Geometric, KLayout (for GDS preprocessing).

### 2. Data Preparation

The repository includes the original ICCAD 2012 GDS files under `data/ICCAD2012/raw/` and pre-defined data splits under `data/ICCAD2012/splits/`.

To generate the derived data representations from the raw GDS files:

#### Point Clouds (for PointNeXt / Hybrid Point)

```bash
python scripts/preprocess/gds2point.py \
    --data_root ./data/ICCAD2012/raw \
    --output_root ./data/ICCAD2012/points_4d \
    --target_points 1024
```

Each output `.npy` file has shape `[1024, 4]` with columns (x, y, n_x, n_y):
- **(x, y)** — coordinates normalized to [0, 1] relative to the 1200×1200 marker window
- **(n_x, n_y)** — outward-facing unit normal vectors

**Sampling strategy:** vertex anchoring → edge interpolation fill → random shuffle.

#### Graphs (for GNN)

```bash
python scripts/preprocess/gds2graph.py \
    --data_root ./data/ICCAD2012/raw \
    --output_root ./data/ICCAD2012/graphs \
    --threshold_nm 65
```

Each output `.pt` file is a PyTorch Geometric `Data` object:
- **Node features** `[N, 4]` — rectangle corners (x1, y1, x2, y2) in [0, 1]
- **Edge index** `[2, E]` — COO format
- **Edge attributes** `[E, 1]` — normalized distance
- **Edge types** `[E, 1]` — 0 = internal (same polygon), 1 = external (cross-polygon)
- **Label** — 1 = hotspot, 0 = non-hotspot

**Graph construction:** Manhattan polygon decomposition → fully-connected internal edges → distance-thresholded external edges (default 65 nm).

#### Image Data (for SimSiam / LPA / Hybrid Image)

Render layout patterns as grayscale PNG images (256×256 or 1200×1200) and place them under the corresponding data directory. Update `config.yaml` paths accordingly.

### 3. Training

Select the model in `config.yaml`:

```yaml
model: "hybrid_point"  # Options: simsiam, lpa, pointnext, hybrid_point, hybrid_image, gnn
```

Then run:

```bash
python train.py
```

Checkpoints and logs are saved to the directory specified in `config.yaml` → `train.save_dir`.

### 4. Inference & Clustering

**Extract embeddings:**

```bash
python inference.py
```

This loads the best checkpoint, extracts L2-normalized features for all test samples, and saves `features.npy` + `filenames.txt`.

**Run clustering:**

```bash
python -m utils.cluster ./results/<experiment_name>
```

Uses adaptive greedy clustering with medoid refinement.

## Reproducing Paper Results

The default `config.yaml` uses **Hybrid PointNeXt** on 4D point clouds. To reproduce the main results:

```bash
# 1. Generate point cloud data
python scripts/preprocess/gds2point.py \
    --data_root ./data/ICCAD2012/raw \
    --output_root ./data/ICCAD2012/points_4d

# 2. Train
python train.py

# 3. Extract features
python inference.py

# 4. Cluster
python -m utils.cluster ./results/hybrid_point
```

## GDS Layer Convention (ICCAD 2012)

| Layer | Purpose |
|:------|:--------|
| 10/0  | Metal1 polygon shapes |
| 21/0  | Hotspot (HS) marker bounding boxes (1200×1200) |
| 23/0  | Non-hotspot (NHS) marker bounding boxes (1200×1200) |

## Citation

If you find this work useful, please cite:

```bibtex
@article{point-jr-2026,
  title   = {Discriminative Joint Representation Learning via High-Fidelity Point Clouds for Pattern Clustering},
  author  = {Longyue Wei, Changzhe Jiao, Rongfang Wang},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

The benchmark data used in this project is from the **ICCAD 2012 CAD Contest**.
