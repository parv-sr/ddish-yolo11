
"""Inspect Fusion predictions at multiple confidence thresholds."""

import argparse
import csv
import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from src.dataset import (
    DDishDataset,
    collate_ddish,
    load_records,
    split_by_source,
    split_random,
)
from src.evaluation.metrics import (
    normalized_boxes_to_xyxy,
    pairwise_box_iou,
)
from src.inference.predictor import DDishPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Fusion detection errors"
    )

    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("random", "source"), default="random"
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/raw")
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Training/validation split seed"
    )
    parser.add_argument(
        "--sample-seed", type=int, default=123,
        help="Which validation patches to inspect"
    )
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.25, 0.35],
    )
    parser.add_argument(
        "--nucleus-conf", type=float, default=0.50
    )
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--zoom", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/fusion_diagnostic"),
    )
    return parser.parse_args()


def has_fusion(record: Mapping[str, Any]) -> bool:
    names = record["names"]
    return any(
        str(names[ann["category_id"]]).lower().startswith("fusion")
        for ann in record["annotations"]
    )


def sample_records(
    records: list[dict[str, Any]],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample roughly 75% Fusion-positive and 25% negative patches."""
    rng = random.Random(seed)

    positive = [r for r in records if has_fusion(r)]
    negative = [r for r in records if not has_fusion(r)]

    n_pos = min(len(positive), math.ceil(count * 0.75))
    n_neg = min(len(negative), count - n_pos)
    n_pos = min(len(positive), count - n_neg)

    selected = (
        rng.sample(positive, n_pos)
        + rng.sample(negative, n_neg)
    )

    if len(selected) != count:
        raise ValueError("Not enough eligible validation images")

    rng.shuffle(selected)
    return selected


def greedy_match(
    pred_boxes: Tensor,
    pred_scores: Tensor,
    gt_boxes: Tensor,
    minimum_iou: float,
) -> dict[int, tuple[int, float]]:
    """
    Match predictions to GT by descending confidence.

    Each GT box may be matched at most once.
    """
    ious = pairwise_box_iou(pred_boxes, gt_boxes)

    matches: dict[int, tuple[int, float]] = {}
    used_gt: set[int] = set()

    order = torch.argsort(
        pred_scores, descending=True
    ).tolist()

    for pred_id in order:
        if len(gt_boxes) == 0:
            break

        candidates = ious[pred_id].clone()

        for gt_id in used_gt:
            candidates[gt_id] = -1

        overlap, gt_index = candidates.max(dim=0)

        if float(overlap) >= minimum_iou:
            gt_id = int(gt_index)
            used_gt.add(gt_id)
            matches[int(pred_id)] = (
                gt_id, float(overlap)
            )

    return matches


def describe_box(box: Tensor) -> str:
    x1, y1, x2, y2 = (
        float(value) for value in box.tolist()
    )
    return (
        f"xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}) "
        f"width={x2-x1:.1f}px height={y2-y1:.1f}px"
    )


def add_mask_overlay(
    image: Image.Image,
    masks: Tensor,
) -> Image.Image:
    layer = Image.new(
        "RGBA", image.size, (0, 0, 0, 0)
    )

    for mask in masks.detach().cpu().numpy():
        alpha = Image.fromarray(
            mask.astype(np.uint8) * 255,
            mode="L",
        )
        layer.paste(
            (45, 145, 245, 65),
            (0, 0),
            alpha,
        )

    return Image.alpha_composite(
        image.convert("RGBA"), layer
    ).convert("RGB")


def dashed_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    color: str,
    dash: int = 12,
) -> None:
    x1, y1, x2, y2 = (round(v) for v in box)

    for x in range(x1, x2, 2 * dash):
        draw.line(
            (x, y1, min(x + dash, x2), y1),
            fill=color, width=2,
        )
        draw.line(
            (x, y2, min(x + dash, x2), y2),
            fill=color, width=2,
        )

    for y in range(y1, y2, 2 * dash):
        draw.line(
            (x1, y, x1, min(y + dash, y2)),
            fill=color, width=2,
        )
        draw.line(
            (x2, y, x2, min(y + dash, y2)),
            fill=color, width=2,
        )


def save_comparison(
    original: Image.Image,
    masks: Tensor,
    gt_boxes: Tensor,
    pred_boxes: Tensor,
    pred_scores: Tensor,
    matches: dict[int, tuple[int, float]],
    threshold: float,
    zoom: int,
    output: Path,
) -> None:
    width, height = original.size
    header = 54
    large_size = (width * zoom, height * zoom)

    left = original.resize(
        large_size, Image.Resampling.NEAREST
    )
    right = add_mask_overlay(
        original, masks
    ).resize(large_size, Image.Resampling.NEAREST)

    canvas = Image.new(
        "RGB",
        (width * zoom * 2, height * zoom + header),
        "white",
    )
    canvas.paste(left, (0, header))
    canvas.paste(right, (width * zoom, header))

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    draw.text(
        (10, 8), "ORIGINAL",
        fill="black", font=font,
    )
    draw.text(
        (width * zoom + 10, 8),
        f"FUSION conf >= {threshold:.3f}",
        fill="black", font=font,
    )
    draw.text(
        (width * zoom + 10, 28),
        "blue nuclei | yellow GT | orange missed | green TP | red FP",
        fill="black", font=font,
    )

    matched_gt = {
        match[0] for match in matches.values()
    }
    x_offset = width * zoom

    # Ground truth: dashed yellow or orange boxes.
    for gt_id, box in enumerate(gt_boxes):
        x1, y1, x2, y2 = (
            float(v) for v in box.tolist()
        )
        scaled = (
            x_offset + x1 * zoom,
            header + y1 * zoom,
            x_offset + x2 * zoom,
            header + y2 * zoom,
        )

        color = (
            "yellow" if gt_id in matched_gt
            else "orange"
        )
        dashed_box(draw, scaled, color)

        draw.text(
            (scaled[0], max(header, scaled[1] - 15)),
            f"G{gt_id}",
            fill=color,
            font=font,
            stroke_width=1,
            stroke_fill="black",
        )

    # Predictions: solid green or red boxes.
    for pred_id, (box, score) in enumerate(
        zip(pred_boxes, pred_scores)
    ):
        x1, y1, x2, y2 = (
            float(v) for v in box.tolist()
        )
        scaled = (
            x_offset + x1 * zoom,
            header + y1 * zoom,
            x_offset + x2 * zoom,
            header + y2 * zoom,
        )

        color = (
            "lime" if pred_id in matches else "red"
        )
        draw.rectangle(
            scaled, outline=color, width=2
        )

        draw.text(
            (
                scaled[0],
                min(
                    header + height * zoom - 14,
                    scaled[3] + 1,
                ),
            ),
            f"P{pred_id}:{float(score):.2f}",
            fill=color,
            font=font,
            stroke_width=1,
            stroke_fill="black",
        )

    output.parent.mkdir(
        parents=True, exist_ok=True
    )
    canvas.save(output)


def write_box_row(
    writer: csv.DictWriter[str],
    image_id: str,
    threshold: float,
    kind: str,
    item_id: int,
    status: str,
    box: Tensor,
    confidence: float | None,
    matched_id: int | None,
    iou: float | None,
) -> None:
    x1, y1, x2, y2 = (
        float(v) for v in box.tolist()
    )

    writer.writerow({
        "image": image_id,
        "threshold": threshold,
        "kind": kind,
        "id": item_id,
        "status": status,
        "confidence": (
            confidence if confidence is not None else ""
        ),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": x2 - x1,
        "height": y2 - y1,
        "matched_id": (
            matched_id if matched_id is not None else ""
        ),
        "iou": iou if iou is not None else "",
    })


def main() -> None:
    args = parse_args()

    if not 5 <= args.count <= 10:
        raise ValueError("--count must be 5 to 10")

    if (
        args.zoom < 1
        or not args.thresholds
        or any(not 0 < t <= 1 for t in args.thresholds)
    ):
        raise ValueError("Invalid zoom or thresholds")

    records = load_records(str(args.data_root))

    if args.split == "random":
        _, validation, split_ids = split_random(
            records, seed=args.seed
        )
    else:
        _, validation, split_ids = split_by_source(
            records, seed=args.seed
        )

    predictor = DDishPredictor(
        args.weights,
        device=args.device,
        conf=min(args.thresholds),
        nms_iou=args.nms_iou,
    )

    saved_split = predictor.metadata.get(
        "split", "source"
    )
    saved_ids = predictor.metadata.get(
        "split_ids",
        predictor.metadata.get("held_out"),
    )

    if (
        saved_split != args.split
        or (
            saved_ids is not None
            and saved_ids != split_ids
        )
    ):
        raise RuntimeError(
            "Checkpoint validation split mismatch"
        )

    chosen = sample_records(
        validation, args.count, args.sample_seed
    )
    dataset = DDishDataset(chosen)

    args.output_dir.mkdir(
        parents=True, exist_ok=True
    )

    csv_path = args.output_dir / "fusion_boxes.csv"

    summary: dict[float, list[int]] = {
        t: [0, 0, 0] for t in args.thresholds
    }

    columns = [
        "image", "threshold", "kind", "id",
        "status", "confidence",
        "x1", "y1", "x2", "y2",
        "width", "height", "matched_id", "iou",
    ]

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns
        )
        writer.writeheader()

        for index in range(len(dataset)):
            batch = collate_ddish([dataset[index]])

            images: Tensor = batch["images"]
            target: dict[str, Any] = (
                batch["targets"][0]
            )

            path = Path(target["path"])
            image_id = (
                f"{path.parent.parent.name}__{path.stem}"
            )

            height, width = images.shape[-2:]

            pixels = (
                images[0]
                .mul(255)
                .clamp(0, 255)
                .byte()
                .permute(1, 2, 0)
                .numpy()
            )
            original = Image.fromarray(
                pixels, mode="RGB"
            )

            # One forward pass per image.
            result = predictor.predict(images)[0]

            # Our Fusion class ID is 0.
            gt_fusion = (
                target["signal_classes"] == 0
            )
            gt_boxes = normalized_boxes_to_xyxy(
                target["signal_boxes"][gt_fusion],
                (height, width),
            ).cpu()

            pred_fusion = (
                result.signals.classes == 0
            )
            boxes = (
                result.signals.boxes[pred_fusion]
                .detach()
                .cpu()
            )
            scores = (
                result.signals.scores[pred_fusion]
                .detach()
                .cpu()
            )

            nuclei = result.nuclei.masks
            if nuclei is None:
                raise RuntimeError(
                    "Missing nucleus segmentation"
                )

            nucleus_filter = (
                result.nuclei.scores
                >= args.nucleus_conf
            )
            selected_masks = nuclei[nucleus_filter]

            print(
                f"\n[{index + 1}/{len(dataset)}] "
                f"{image_id}"
            )
            print(
                f"  GT Fusion: {len(gt_boxes)} | "
                f"nuclei displayed: {len(selected_masks)}"
            )

            for gt_id, box in enumerate(gt_boxes):
                print(
                    f"    GT {gt_id}: "
                    f"{describe_box(box)}"
                )

            for threshold in args.thresholds:
                keep = scores >= threshold
                pred_boxes = boxes[keep]
                pred_scores = scores[keep]

                matches = greedy_match(
                    pred_boxes,
                    pred_scores,
                    gt_boxes,
                    args.match_iou,
                )

                matched_gt = {
                    gt for gt, _ in matches.values()
                }

                tp = len(matches)
                fp = len(pred_boxes) - tp
                fn = len(gt_boxes) - tp

                summary[threshold][0] += tp
                summary[threshold][1] += fp
                summary[threshold][2] += fn

                print(
                    f"  threshold {threshold:.3f}: "
                    f"TP={tp}, FP={fp}, FN={fn}"
                )

                for pred_id, (box, score) in enumerate(
                    zip(pred_boxes, pred_scores)
                ):
                    matched = matches.get(pred_id)
                    status = (
                        "TP" if matched else "FP"
                    )

                    message = (
                        f"    P{pred_id} {status} "
                        f"conf={float(score):.3f} "
                        f"{describe_box(box)}"
                    )

                    if matched:
                        message += (
                            f" matches G{matched[0]}, "
                            f"IoU={matched[1]:.3f}"
                        )

                    print(message)

                    write_box_row(
                        writer,
                        image_id,
                        threshold,
                        "prediction",
                        pred_id,
                        status,
                        box,
                        float(score),
                        matched[0] if matched else None,
                        matched[1] if matched else None,
                    )

                for gt_id, box in enumerate(gt_boxes):
                    is_matched = gt_id in matched_gt

                    if not is_matched:
                        print(
                            f"    MISSED G{gt_id}: "
                            f"{describe_box(box)}"
                        )

                    write_box_row(
                        writer,
                        image_id,
                        threshold,
                        "ground_truth",
                        gt_id,
                        "MATCHED" if is_matched else "FN",
                        box,
                        None,
                        None,
                        None,
                    )

                folder = (
                    args.output_dir
                    / f"conf_{threshold:.3f}"
                )

                save_comparison(
                    original,
                    selected_masks,
                    gt_boxes,
                    pred_boxes,
                    pred_scores,
                    matches,
                    threshold,
                    args.zoom,
                    folder / f"{image_id}.png",
                )

    print("\nDIAGNOSTIC SUMMARY (selected patches only)")

    for threshold, (tp, fp, fn) in summary.items():
        precision = (
            tp / (tp + fp) if tp + fp else 0.0
        )
        recall = (
            tp / (tp + fn) if tp + fn else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        print(
            f"  {threshold:.3f}: "
            f"TP={tp} FP={fp} FN={fn} "
            f"P={precision:.3f} "
            f"R={recall:.3f} "
            f"F1={f1:.3f}"
        )

    print(
        f"\nSaved PNGs and boxes CSV under: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()

