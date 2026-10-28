"""Evaluate the dual-head model on its saved validation split."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import torch
from torch.utils.data import DataLoader

from src.dataset import (
    DDishDataset,
    collate_ddish,
    load_records,
    split_by_source,
    split_random,
)

from src.evaluation.metrics import (
    APAccumulator,
    normalized_boxes_to_xyxy,
    pairwise_box_iou,
    pairwise_mask_iou,
)

from src.inference.predictor import DDishPredictor


SIGNAL_NAMES = (
    "Fusion",
    "Chr17 (pink)",
    "HER2 (black)",
)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Evaluate a dual-head D-DISH checkpoint"
    )

    parser.add_argument(
        "--weights", type=Path, required=True
    )

    parser.add_argument(
        "--split",
        choices=("random", "source"),
        required=True,
    )

    parser.add_argument(
        "--seed", type=int, default=42
    )

    parser.add_argument(
        "--batch-size", type=int, default=4
    )

    parser.add_argument(
        "--device", type=str, default=None
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Minimum confidence retained for AP",
    )

    parser.add_argument(
        "--fixed-conf",
        type=float,
        default=0.25,
        help="Confidence threshold for P/R/F1",
    )

    parser.add_argument(
        "--nms-iou", type=float, default=0.7
    )

    parser.add_argument(
        "--max-det", type=int, default=300
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/evaluation"),
    )

    return parser.parse_args()


def format_percent(value: object) -> str:
    if value is None:
        return "N/A"

    return f"{100 * float(value):.2f}%"


def print_table(
    title: str,
    results: dict[str, Any],
) -> None:

    print(f"\n{title}")

    print(
        f"{'Class':<18} "
        f"{'GT':>6} {'TP':>6} {'FP':>6} {'FN':>6} "
        f"{'P':>9} {'R':>9} {'F1':>9} "
        f"{'AP50':>9} {'AP50-95':>9}"
    )

    for row in results["classes"]:
        print(
            f"{row['class']:<18} "
            f"{row['gt']:>6} "
            f"{row['tp']:>6} "
            f"{row['fp']:>6} "
            f"{row['fn']:>6} "
            f"{format_percent(row['precision']):>9} "
            f"{format_percent(row['recall']):>9} "
            f"{format_percent(row['f1']):>9} "
            f"{format_percent(row['ap50']):>9} "
            f"{format_percent(row['map50_95']):>9}"
        )

    macro = results["macro"]

    print(
        "Macro avg:"
        + "".join(
            f" {format_percent(macro[key]):>9}"
            for key in (
                "precision",
                "recall",
                "f1",
                "ap50",
                "map50_95",
            )
        )
    )

    if "matched_instance_dice_at_iou50" in results:
        print(
            "Matched-instance Dice "
            "(IoU >= 0.50, fixed confidence): "
            f"{results['matched_instance_dice_at_iou50']:.4f} "
            f"over {results['matched_instance_count']} nuclei"
        )


def save_csv(
    path: Path,
    results: dict[str, Any],
) -> None:

    rows: list[dict[str, Any]] = results["classes"]

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

def print_confidence_sweep(
    metric: APAccumulator,
    thresholds: tuple[float, ...] = (
        0.001, 0.05, 0.10, 0.15,
        0.20, 0.25, 0.35, 0.50,
    ),
) -> None:
    """Report per-class P/R/F1 at different confidence levels."""

    for class_id, class_name in enumerate(metric.class_names):
        scores: np.ndarray = np.concatenate(
            metric.scores[class_id]
        )

        correct: np.ndarray = np.concatenate(
            metric.matches[class_id],
            axis=0,
        )

        n_gt: int = int(metric.gt_count[class_id])

        print(f"\n{class_name}")
        print(
            f"{'Conf':>7} {'Pred':>7} {'TP':>7} "
            f"{'P':>9} {'R':>9} {'F1':>9}"
        )

        for threshold in thresholds:
            selected: np.ndarray = scores >= threshold

            n_pred: int = int(selected.sum())
            tp: int = int(correct[selected, 0].sum())
            fp: int = n_pred - tp

            precision: float = (
                tp / (tp + fp) if tp + fp else 0.0
            )

            recall: float = (
                tp / n_gt if n_gt else 0.0
            )

            f1: float = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )

            print(
                f"{threshold:7.3f} "
                f"{n_pred:7d} "
                f"{tp:7d} "
                f"{precision:9.3f} "
                f"{recall:9.3f} "
                f"{f1:9.3f}"
            )


def main() -> None:

    args = parse_args()

    if not (0 <= args.conf <= args.fixed_conf <= 1):
        raise ValueError(
            "Require 0 <= --conf <= --fixed-conf <= 1"
        )

    predictor = DDishPredictor(
        args.weights,
        device=args.device,
        conf=args.conf,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
    )

    records = load_records("data/raw")

    if args.split == "random":
        _, validation, split_ids = split_random(
            records,
            seed=args.seed,
        )
    else:
        _, validation, split_ids = split_by_source(
            records,
            seed=args.seed,
        )

    # Prevent accidental evaluation using the wrong split.
    checkpoint_split = predictor.metadata.get(
        "split", "source"
    )

    checkpoint_ids = predictor.metadata.get(
        "split_ids",
        predictor.metadata.get("held_out"),
    )

    if (
        checkpoint_split != args.split
        or checkpoint_ids != split_ids
    ):
        raise ValueError(
            "Validation split differs from saved checkpoint"
        )

    loader = DataLoader(
        DDishDataset(validation),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_ddish,
    )

    signal_metric = APAccumulator(
        SIGNAL_NAMES,
        fixed_conf=args.fixed_conf,
    )

    nucleus_metric = APAccumulator(
        ("Nucleus",),
        fixed_conf=args.fixed_conf,
    )

    for batch_number, batch in enumerate(
        loader, start=1
    ):

        images: torch.Tensor = batch["images"]
        predictions = predictor.predict(images)

        height, width = images.shape[-2:]

        for pred, target in zip(
            predictions,
            batch["targets"],
        ):

            # Gene-signal bounding-box evaluation
            gt_signal_boxes = normalized_boxes_to_xyxy(
                target["signal_boxes"].to(predictor.device),
                (height, width),
            )

            gt_signal_classes = target[
                "signal_classes"
            ].to(predictor.device)

            signal_ious = pairwise_box_iou(
                pred.signals.boxes,
                gt_signal_boxes,
            )

            signal_metric.update(
                pred.signals.scores,
                pred.signals.classes,
                gt_signal_classes,
                signal_ious,
            )

            # Nucleus instance-mask evaluation
            gt_masks = target[
                "nucleus_masks"
            ].to(predictor.device)

            if pred.nuclei.masks is None:
                raise RuntimeError(
                    "Segmentation output contains no masks"
                )

            mask_ious = pairwise_mask_iou(
                pred.nuclei.masks,
                gt_masks,
            )

            gt_nucleus_classes = torch.zeros(
                len(gt_masks),
                device=predictor.device,
                dtype=torch.long,
            )

            nucleus_metric.update(
                pred.nuclei.scores,
                pred.nuclei.classes,
                gt_nucleus_classes,
                mask_ious,
                track_dice=True,
            )

        if (
            batch_number % 5 == 0
            or batch_number == len(loader)
        ):
            print(
                f"Evaluated {batch_number}/{len(loader)} batches",
                flush=True,
            )

    signal_results = signal_metric.summarize()
    nucleus_results = nucleus_metric.summarize()

    print_confidence_sweep(signal_metric)
    print_confidence_sweep(nucleus_metric)

    print_table(
        "GENE SIGNAL DETECTION (boxes)",
        signal_results,
    )

    print_table(
        "NUCLEUS INSTANCE SEGMENTATION (masks)",
        nucleus_results,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report: dict[str, Any] = {
        "weights": str(args.weights),
        "split": args.split,
        "seed": args.seed,
        "validation_images": len(validation),
        "minimum_ap_conf": args.conf,
        "fixed_prf1_conf": args.fixed_conf,
        "nms_iou": args.nms_iou,
        "match_iou": 0.50,
        "signal_detection": signal_results,
        "nucleus_segmentation": nucleus_results,
    }

    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2)
    )

    save_csv(
        args.output_dir / "signals.csv",
        signal_results,
    )

    save_csv(
        args.output_dir / "nuclei.csv",
        nucleus_results,
    )

    print("\nSaved metrics to", args.output_dir)


if __name__ == "__main__":
    main()
