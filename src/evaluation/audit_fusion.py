"""Audit Fusion localization, class competition, confidence, and NMS survival."""

import argparse
import csv
import inspect
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from ultralytics.utils.ops import xywh2xyxy

from src.dataset import DDishDataset, collate_ddish
from src.evaluation.metrics import normalized_boxes_to_xyxy, pairwise_box_iou
from src.evaluation.validation import greedy_matches, validation_records
from src.inference.postprocess import non_max_suppression
from src.inference.predictor import DDishPredictor

NAMES: tuple[str, ...] = ("Fusion", "Chr17", "HER2")


def audit_image(raw: Tensor, final: Tensor, kept_ids: Tensor, gt_boxes: Tensor, gt_classes: Tensor, threshold: float, floor: float, match_iou: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if raw.ndim != 2 or raw.shape[0] != 7 or final.ndim != 2 or final.shape[1] < 6:
        raise ValueError("Expected decoded raw (7, A) and NMS rows (N, 6+)")

    boxes: Tensor = xywh2xyxy(raw[:4].T)
    scores: Tensor = raw[4:7].T
    confidence, winners = scores.max(dim=1)
    fusion_gt: Tensor = gt_boxes[gt_classes == 0]
    fusion_final: Tensor = final[(final[:, 5] == 0) & (final[:, 4] >= threshold)]
    final_ious: Tensor = pairwise_box_iou(fusion_final[:, :4], fusion_gt)
    matches: dict[int, int] = greedy_matches(final_ious, fusion_final[:, 4], match_iou)
    matched_gt: set[int] = set(matches.values())
    raw_ious: Tensor = pairwise_box_iou(boxes, fusion_gt)
    kept: Tensor = torch.zeros(len(scores), dtype=torch.bool)
    kept[kept_ids.reshape(-1).long()] = True
    instances: list[dict[str, Any]] = []

    for index, gt_box in enumerate(fusion_gt):
        localized: Tensor = raw_ious[:, index] >= match_iou
        floor_candidates: Tensor = localized & (confidence >= floor)
        confident: Tensor = localized & (confidence >= threshold)
        fusion_wins: Tensor = confident & (winners == 0)
        other_wins: Tensor = confident & (winners != 0)

        if index in matched_gt:
            status: str = "matched"
        elif bool((final_ious[:, index] >= match_iou).any()):
            status = "matched_elsewhere"
        elif bool(fusion_wins.any()):
            status = "suppressed_or_capped"
        elif bool(other_wins.any()):
            status = "other_class_wins"
        elif bool(floor_candidates.any()):
            status = "below_threshold"
        elif bool(localized.any()):
            status = "below_floor"
        else:
            status = "no_localized_candidate"

        candidates: Tensor = torch.where(localized)[0]
        representative: int | None = int(candidates[scores[candidates, 0].argmax()]) if len(candidates) else None
        instances.append({
            "threshold": threshold,
            "gt_fusion_index": index,
            "status": status,
            "gt_x1": float(gt_box[0]),
            "gt_y1": float(gt_box[1]),
            "gt_x2": float(gt_box[2]),
            "gt_y2": float(gt_box[3]),
            "gt_width": float(gt_box[2] - gt_box[0]),
            "gt_height": float(gt_box[3] - gt_box[1]),
            "max_raw_iou": float(raw_ious[:, index].max()) if len(raw_ious) else 0.0,
            "localized_candidates": int(localized.sum()),
            "confident_fusion_winners": int(fusion_wins.sum()),
            "confident_other_winners": int(other_wins.sum()),
            "candidate_index": representative,
            "candidate_iou": float(raw_ious[representative, index]) if representative is not None else None,
            "candidate_fusion_score": float(scores[representative, 0]) if representative is not None else None,
            "candidate_chr17_score": float(scores[representative, 1]) if representative is not None else None,
            "candidate_her2_score": float(scores[representative, 2]) if representative is not None else None,
            "candidate_winner": NAMES[int(winners[representative])] if representative is not None else None,
            "candidate_survived_nms": bool(kept[representative]) if representative is not None else None,
        })

    all_ious: Tensor = pairwise_box_iou(fusion_final[:, :4], gt_boxes)
    false_positives: list[dict[str, Any]] = []

    for index, detection in enumerate(fusion_final):
        if index in matches:
            continue

        best_index: int | None = int(all_ious[index].argmax()) if len(gt_boxes) else None
        overlap: float = float(all_ious[index, best_index]) if best_index is not None else 0.0
        label: str | None = NAMES[int(gt_classes[best_index])] if best_index is not None else None
        category: str = "overlaps_fusion_gt_already_claimed" if overlap >= match_iou and label == "Fusion" else (
            "overlaps_other_signal_gt" if overlap >= match_iou else "no_matching_annotated_signal"
        )

        false_positives.append({
            "threshold": threshold,
            "confidence": float(detection[4]),
            "x1": float(detection[0]),
            "y1": float(detection[1]),
            "x2": float(detection[2]),
            "y2": float(detection[3]),
            "best_gt_iou": overlap,
            "best_gt_class": label,
            "category": category,
        })

    return instances, false_positives


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", type=float, nargs="+", default=(0.10, 0.15, 0.25))
    parser.add_argument("--floor", type=float, default=0.001)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args: argparse.Namespace = parse_args()
    thresholds: list[float] = sorted(set(args.thresholds))

    if not thresholds or not 0 < args.floor <= min(thresholds) <= max(thresholds) <= 1:
        raise ValueError("Invalid confidence floor or audit thresholds")
    if not 0 < args.match_iou <= 1 or not 0 < args.nms_iou <= 1 or args.max_det < 1:
        raise ValueError("Invalid IoU threshold or max detections")
    if "return_idxs" not in inspect.signature(non_max_suppression).parameters:
        raise RuntimeError("Installed Ultralytics NMS lacks return_idxs; update the environment deliberately")

    predictor = DDishPredictor(args.weights, device=args.device, conf=args.floor, nms_iou=args.nms_iou, max_det=args.max_det)
    model = predictor.model

    if model.signal_head.end2end or model.signal_head.xyxy:
        raise RuntimeError("Audit supports only the standard non-end-to-end xywh signal head")

    validation = validation_records(predictor.metadata, args.data_root, args.seed)
    loader = DataLoader(DDishDataset(validation), batch_size=1, collate_fn=collate_ddish, num_workers=0)
    instances: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    summary: dict[float, Counter[str]] = {threshold: Counter() for threshold in thresholds}
    fp_summary: dict[float, Counter[str]] = {threshold: Counter() for threshold in thresholds}
    capped_images: int = 0

    for image_index, batch in enumerate(loader, start=1):
        target: dict[str, Any] = batch["targets"][0]
        image: Tensor = batch["images"].to(predictor.device)
        raw: Tensor = model(image)["signals"][0]

        if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != 7 or not bool(torch.isfinite(raw).all()):
            raise RuntimeError(f"Unexpected decoded signal predictions: {tuple(raw.shape)}")

        final, indices = non_max_suppression(raw.clone(), conf_thres=args.floor, iou_thres=args.nms_iou, nc=3, max_det=args.max_det, multi_label=False, agnostic=False, return_idxs=True)
        capped_images += int(len(final[0]) == args.max_det)
        shape: tuple[int, int] = tuple(image.shape[-2:])
        gt_boxes: Tensor = normalized_boxes_to_xyxy(target["signal_boxes"], shape).cpu()
        gt_classes: Tensor = target["signal_classes"].cpu()

        for threshold in thresholds:
            found, spurious = audit_image(raw[0].detach().cpu(), final[0].detach().cpu(), indices[0].detach().cpu(), gt_boxes, gt_classes, threshold, args.floor, args.match_iou)

            for row in found:
                row.update(image=str(target["path"]), source_group=str(target["source_group"]))
                summary[threshold][row["status"]] += 1
                instances.append(row)

            for row in spurious:
                row.update(image=str(target["path"]), source_group=str(target["source_group"]))
                fp_summary[threshold][row["category"]] += 1
                false_positives.append(row)

        if image_index % 20 == 0 or image_index == len(loader):
            print(f"Fusion audit: {image_index}/{len(loader)} images", flush=True)

    report: dict[str, Any] = {
        "weights": str(args.weights),
        "validation_images": len(validation),
        "confidence_floor": args.floor,
        "thresholds": thresholds,
        "match_iou": args.match_iou,
        "nms_iou": args.nms_iou,
        "statuses": {str(threshold): dict(counts) for threshold, counts in summary.items()},
        "false_positive_categories": {str(threshold): dict(counts) for threshold, counts in fp_summary.items()},
        "potentially_capped_images": capped_images,
        "interpretation": "Heuristic stage attribution using raw decoded anchors, best-class selection, and NMS. Suppressed_or_capped combines NMS suppression and candidate limits; overlap does not prove biological identity.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "fusion_instances.csv", instances)
    write_csv(args.output_dir / "fusion_false_positives.csv", false_positives)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
