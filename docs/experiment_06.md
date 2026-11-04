# Experiment 06 — YOLO11s capacity test

## Hypothesis

Increasing backbone/neck capacity from YOLO11n to YOLO11s improves Fusion detection without changing input resolution or supervision. The classifier-transfer variant stays enabled, and the copied classifier trunks remain trainable.

## Controls

- Same 720/180 random patch split, seed 42, as Experiment 05.
- Same 256 x 256 patches, P2 signal detection, geometric augmentation, 50 epochs, AdamW, cosine schedule and learning rate 1e-4.
- Only the model size changes from `n` to `s`, including its matching pretrained YOLO11s initialization.
- No P2 nucleus segmentation or U-Net changes.
- Compare on the same 180 images, confidence and IoU thresholds.

## Smoke tests

```bash
uv run python -m unittest discover -s tests -v
uv run python -m scripts.smoke_train --model classifier_transfer --size s --p2-detection
```

## Training

```bash
uv run python -m src.train \
  --model classifier_transfer --size s --epochs 50 --batch-size 8 --lr 1e-4 \
  --split random --p2-detection --geom-augment \
  --run-dir runs/experiment_06_yolo11s
```

Do not resume the YOLO11n Experiment 05 checkpoint: parameter shapes differ. If the A4500 runs out of memory, try `--batch-size 4`, record that change, and recognize that different batch size is a confounding variable.

## Detection and segmentation evaluation

```bash
for checkpoint in best last; do
  uv run python -m src.evaluation.evaluate \
    --weights "runs/experiment_06_yolo11s/${checkpoint}.pt" \
    --split random --conf 0.001 --fixed-conf 0.25 --nms-iou 0.70 \
    --output-dir "runs/experiment_06_yolo11s/evaluation_${checkpoint}"
done
```

## Paired counting and Fusion auditing

Run the same diagnostics on the Experiment 05 and Experiment 06 `last.pt` checkpoints:

```bash
for run in experiment_05_transfer experiment_06_yolo11s; do
  uv run python -m src.evaluation.counts \
    --weights "runs/${run}/last.pt" \
    --signal-conf 0.25 --nucleus-conf 0.50 --match-iou 0.50 --nms-iou 0.70 \
    --output-dir "runs/${run}/counting_last"

  uv run python -m src.evaluation.audit_fusion \
    --weights "runs/${run}/last.pt" \
    --thresholds 0.10 0.15 0.25 --floor 0.001 --match-iou 0.50 --nms-iou 0.70 \
    --output-dir "runs/${run}/fusion_audit_last"
done
```

Choose new suffixes if output directories exist. The audit uses class-aware NMS with `return_idxs` to trace raw decoded candidates, and it flags potentially capped images.

## Readout

1. Primary: Fusion AP50/AP50-95, Fusion confidence sweep F1, and signal macro mAP50.
2. Secondary: nucleus mask AP50-95 and matched-instance Dice.
3. Audit: other-class winners, unlocalized candidates, low-confidence candidates, and suppressed/capped candidates; false-positive overlap categories.
4. Counts: per-reference-nucleus MAE and bias using oracle and predicted nucleus masks; matched-only counts, missed/extra nuclei and patch-level count error.

Signals are counted only when their box centers lie inside exactly one nucleus mask. Fusion contributes once to HER2 and once to Chr17 in derived per-nucleus totals. Outside and ambiguous signals are excluded and tallied. These are research counts, not clinical copy-number interpretations.

The random patch split can contain related patches from the same source field, so these validation results are not an independent deployment-test estimate.
