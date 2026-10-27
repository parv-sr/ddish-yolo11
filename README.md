# D-DISH YOLO11

Unified multitask YOLO11 for D-DISH breast cancer microscopy.

A pretrained YOLO11 backbone and neck are shared between:
- Nucleus instance segmentation (1 class)
- HER2, Chr17 and Fusion signal detection (3 classes)

The current YOLO11n configuration has **3,274,060 parameters**.

## Setup

```bash
uv sync
```

Dataset layout:

```text
data/raw/
  503610SL-SNG/
  503674SL-SNG/
  503763SL-SNG/
```

Each dataset contains `patches/` and `initial_annotations/instances_default.json`. Medical image data and trained weights are not committed to Git.

## Training

Run from the repository root:

```bash
# SL-grouped validation
uv run python -m src.train --epochs 30 --batch-size 8 \
  --split source --run-dir runs/experiment_01

# Random 80/20 validation
uv run python -m src.train --epochs 30 --batch-size 8 \
  --lr 1e-4 --split random --run-dir runs/experiment_02_random

# Resume an interrupted experiment
uv run python -m src.train --epochs 30 --batch-size 8 \
  --split random --run-dir runs/experiment_02_random --resume
```

Key arguments: `--epochs`, `--batch-size`, `--lr`, `--workers`, `--split`, `--run-dir`, `--resume`.

Training produces `best.pt`, `last.pt` and `history.csv` under the run directory.

## Evaluation

```bash
uv run python -m src.evaluation.evaluate \
  --weights runs/experiment_02_random/best.pt \
  --split random \
  --output-dir runs/experiment_02_random/evaluation
```

Reports class-wise precision, recall, F1, AP50 and AP50–95, along with nucleus instance-segmentation metrics.

## References

- Ultralytics. [YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/).
- Sun, Y., & Liang, Z. (2026). *Tiny fabric defect detection with a high resolution branch in YOLOv11n.* Discover Artificial Intelligence, 6, Article 1574. https://doi.org/10.1007/s44163-026-02553-w

## Research directions

- P2/stride-4 prediction branch for tiny gene signals
- Microscopy-specific data augmentation and pretraining
- Nucleus-wise HER2/Chr17 signal quantification

