"""Evaluate nucleus-assigned signal counts with oracle and predicted masks."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.associate import AssociationReport, associate_signals
from src.dataset import DDishDataset, collate_ddish
from src.evaluation.metrics import normalized_boxes_to_xyxy, pairwise_mask_iou
from src.evaluation.validation import greedy_matches, validation_records
from src.inference.postprocess import Detections, PatchPredictions
from src.inference.predictor import DDishPredictor

COUNT_NAMES: tuple[str, ...] = ("fusion", "chr17", "her2", "her2_total", "chr17_total")


def count_vectors(report: AssociationReport) -> Tensor:
    raw: Tensor = torch.tensor([[n.fusion, n.chr17, n.her2] for n in report.nuclei], dtype=torch.float64).reshape(-1, 3)
    return torch.cat((raw, raw[:, 0:1] + raw[:, 2:3], raw[:, 0:1] + raw[:, 1:2]), dim=1)


class CountMetric:
    def __init__(self) -> None:
        self.n: int = 0
        self.absolute: Tensor = torch.zeros(len(COUNT_NAMES), dtype=torch.float64)
        self.signed: Tensor = torch.zeros(len(COUNT_NAMES), dtype=torch.float64)
        self.exact: Tensor = torch.zeros(len(COUNT_NAMES), dtype=torch.float64)

    def update(self, errors: Tensor) -> None:
        if errors.ndim != 2 or errors.shape[1] != len(COUNT_NAMES):
            raise ValueError("Expected a matrix of per-class count errors")

        self.n += len(errors)
        self.absolute += errors.abs().sum(dim=0)
        self.signed += errors.sum(dim=0)
        self.exact += (errors == 0).sum(dim=0)

    def summary(self) -> dict[str, Any]:
        return {
            "samples": self.n,
            "classes": {name: {
                "mae": float(self.absolute[i] / self.n) if self.n else None,
                "bias": float(self.signed[i] / self.n) if self.n else None,
                "exact_fraction": float(self.exact[i] / self.n) if self.n else None,
            } for i, name in enumerate(COUNT_NAMES)},
        }


def compare_image(prediction: PatchPredictions, target: dict[str, Any], signal_conf: float, nucleus_conf: float, match_iou: float) -> dict[str, Any]:
    masks: Tensor = target["nucleus_masks"].cpu().bool()
    shape: tuple[int, int] = tuple(masks.shape[-2:])
    boxes: Tensor = normalized_boxes_to_xyxy(target["nucleus_boxes"].cpu(), shape)

    oracle_nuclei = Detections(boxes=boxes, scores=torch.ones(len(masks)), classes=torch.zeros(len(masks), dtype=torch.long), masks=masks)
    oracle_signals = Detections(
        boxes=normalized_boxes_to_xyxy(target["signal_boxes"].cpu(), shape),
        scores=torch.ones(len(target["signal_boxes"])),
        classes=target["signal_classes"].cpu(),
    )

    truth: AssociationReport = associate_signals(PatchPredictions(nuclei=oracle_nuclei, signals=oracle_signals), nucleus_conf=0, signal_conf=0)
    oracle: AssociationReport = associate_signals(PatchPredictions(nuclei=oracle_nuclei, signals=prediction.signals), nucleus_conf=0, signal_conf=signal_conf)
    pipeline: AssociationReport = associate_signals(prediction, nucleus_conf=nucleus_conf, signal_conf=signal_conf)

    expected: Tensor = count_vectors(truth)
    oracle_counts: Tensor = count_vectors(oracle)
    predicted: Tensor = count_vectors(pipeline)
    retained: list[int] = [n.nucleus_index for n in pipeline.nuclei]

    if prediction.nuclei.masks is None:
        raise RuntimeError("Predicted nucleus masks are missing")

    overlaps: Tensor = pairwise_mask_iou(prediction.nuclei.masks[retained].detach().cpu().bool(), masks)
    matches: dict[int, int] = greedy_matches(overlaps, prediction.nuclei.scores[retained].detach().cpu(), match_iou)
    aligned: Tensor = torch.zeros_like(expected)

    for predicted_index, gt_index in matches.items():
        aligned[gt_index] = predicted[predicted_index]

    matched_gt: set[int] = set(matches.values())
    extra: list[int] = [index for index in range(len(predicted)) if index not in matches]
    records: list[dict[str, Any]] = []

    for gt_index in range(len(expected)):
        row: dict[str, Any] = {
            "image": str(target["path"]),
            "source_group": str(target["source_group"]),
            "gt_nucleus": gt_index,
            "matched": gt_index in matched_gt,
        }

        for class_index, name in enumerate(COUNT_NAMES):
            row[f"gt_{name}"] = int(expected[gt_index, class_index])
            row[f"oracle_{name}"] = int(oracle_counts[gt_index, class_index])
            row[f"pred_{name}"] = int(aligned[gt_index, class_index])

        records.append(row)

    gt_error: Tensor = aligned - expected
    matched_indices: list[int] = sorted(matched_gt)

    return {
        "rows": records,
        "errors": {
            "oracle_masks_per_gt": oracle_counts - expected,
            "predicted_masks_per_gt": gt_error,
            "predicted_masks_matched_only": gt_error[matched_indices],
            "predicted_masks_patch_totals": (predicted.sum(dim=0) - expected.sum(dim=0)).unsqueeze(0),
        },
        "coverage": {
            "gt_nuclei": len(expected),
            "matched_nuclei": len(matches),
            "missed_nuclei": len(expected) - len(matches),
            "extra_predicted_nuclei": len(extra),
        },
        "extra_counts": predicted[extra].sum(dim=0),
        "association": {
            "ground_truth": truth.status_counts,
            "oracle_masks": oracle.status_counts,
            "predicted_masks": pipeline.status_counts,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--signal-conf", type=float, default=0.25)
    parser.add_argument("--nucleus-conf", type=float, default=0.50)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args: argparse.Namespace = parse_args()

    if not (0.0 < min(args.signal_conf, args.nucleus_conf) <= max(args.signal_conf, args.nucleus_conf) <= 1.0):
        raise ValueError("Confidence thresholds must be in (0, 1]")
    if not 0 < args.match_iou <= 1 or not 0 < args.nms_iou <= 1 or args.max_det < 1:
        raise ValueError("Invalid IoU threshold or max detections")

    predictor = DDishPredictor(args.weights, device=args.device, conf=min(args.signal_conf, args.nucleus_conf), nms_iou=args.nms_iou, max_det=args.max_det)
    validation = validation_records(predictor.metadata, args.data_root, args.seed)
    loader = DataLoader(DDishDataset(validation), batch_size=1, collate_fn=collate_ddish, num_workers=0)
    metrics: dict[str, CountMetric] = {key: CountMetric() for key in ("oracle_masks_per_gt", "predicted_masks_per_gt", "predicted_masks_matched_only", "predicted_masks_patch_totals")}
    coverage: Counter[str] = Counter()
    association: dict[str, Counter[str]] = {key: Counter() for key in ("ground_truth", "oracle_masks", "predicted_masks")}
    extra_counts: Tensor = torch.zeros(len(COUNT_NAMES), dtype=torch.float64)
    rows: list[dict[str, Any]] = []
    capped_images: int = 0

    for index, batch in enumerate(loader, start=1):
        prediction: PatchPredictions = predictor.predict(batch["images"])[0]
        result: dict[str, Any] = compare_image(prediction, batch["targets"][0], args.signal_conf, args.nucleus_conf, args.match_iou)
        rows.extend(result["rows"])
        coverage.update(result["coverage"])
        extra_counts += result["extra_counts"]
        capped_images += int(len(prediction.signals.scores) == args.max_det or len(prediction.nuclei.scores) == args.max_det)

        for name, errors in result["errors"].items():
            metrics[name].update(errors)

        for name, counts in result["association"].items():
            association[name].update(counts)

        if index % 20 == 0 or index == len(loader):
            print(f"Counting: {index}/{len(loader)} images", flush=True)

    report: dict[str, Any] = {
        "weights": str(args.weights),
        "validation_images": len(validation),
        "signal_conf": args.signal_conf,
        "nucleus_conf": args.nucleus_conf,
        "match_iou": args.match_iou,
        "nms_iou": args.nms_iou,
        "count_metrics": {key: value.summary() for key, value in metrics.items()},
        "nucleus_coverage": dict(coverage),
        "association_status": {key: dict(value) for key, value in association.items()},
        "extra_predicted_nucleus_counts": dict(zip(COUNT_NAMES, extra_counts.long().tolist())),
        "potentially_capped_images": capped_images,
        "definitions": {
            "her2_total": "HER2 + Fusion",
            "chr17_total": "Chr17 + Fusion",
            "oracle_masks_per_gt": "Predicted signals assigned to reference nucleus masks",
            "predicted_masks_per_gt": "Predicted signals assigned to matched predicted masks; missed reference nuclei count as zero",
            "predicted_masks_matched_only": "Only successfully matched reference nuclei",
            "predicted_masks_patch_totals": "All predicted nuclei, including unmatched extras, compared with summed reference counts",
            "excluded": "Outside-mask and multiply-assigned signals are reported but not counted",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if rows:
        with (args.output_dir / "nuclei.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
