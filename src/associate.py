"""Assign gene-signal detections to nucleus instances."""

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from src.inference.postprocess import PatchPredictions
from src.inference.predictor import DDishPredictor


SIGNAL_NAMES: tuple[str, str, str] = (
    "Fusion", "Chr17", "HER2"
)

AssociationStatus = Literal[
    "assigned", "outside", "ambiguous"
]


@dataclass(frozen=True)
class SignalAssignment:
    signal_index: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    center_xy: tuple[float, float]
    status: AssociationStatus
    nucleus_index: int | None
    candidate_nuclei: tuple[int, ...]


@dataclass(frozen=True)
class NucleusCount:
    nucleus_index: int
    confidence: float
    fusion: int
    chr17: int
    her2: int


@dataclass(frozen=True)
class AssociationReport:
    nuclei: list[NucleusCount]
    signals: list[SignalAssignment]
    status_counts: dict[str, int]


def associate_signals(
    prediction: PatchPredictions,
    *,
    nucleus_conf: float = 0.50,
    signal_conf: float = 0.25,
) -> AssociationReport:
    """
    Assign each retained signal to exactly one nucleus.

    Signals outside retained masks or inside overlapping
    masks are excluded from nucleus-wise counts but
    preserved in the report.
    """
    if not (0.0 <= nucleus_conf <= 1.0 and 0.0 <= signal_conf <= 1.0):
        raise ValueError("Confidence thresholds must be between 0 and 1")

    if prediction.nuclei.masks is None:
        raise ValueError("Nucleus masks are required")

    masks: Tensor = prediction.nuclei.masks.detach().cpu().bool()

    nucleus_scores: Tensor = prediction.nuclei.scores.detach().cpu()

    signal_scores: Tensor = prediction.signals.scores.detach().cpu()

    signal_boxes: Tensor = prediction.signals.boxes.detach().cpu()

    signal_classes: Tensor = prediction.signals.classes.detach().cpu().long()

    if (masks.ndim != 3 or masks.shape[0] != len(nucleus_scores)):
        raise ValueError("Nucleus mask and confidence dimensions disagree")

    height, width = masks.shape[-2:]

    nucleus_indices: list[int] = torch.where(nucleus_scores >= nucleus_conf)[0].tolist()

    counts: dict[int, list[int]] = {
        idx: [0, 0, 0] for idx in nucleus_indices
    }

    assignments: list[SignalAssignment] = []
    statuses: Counter[str] = Counter()

    for signal_idx in range(len(signal_scores)):

        confidence: float = float(signal_scores[signal_idx])

        if confidence < signal_conf:
            continue

        class_idx: int = int(
            signal_classes[signal_idx]
        )

        if not 0 <= class_idx < len(SIGNAL_NAMES):
            raise ValueError(
                f"Unknown signal class: {class_idx}"
            )

        x1, y1, x2, y2 = (
            float(value)
            for value in signal_boxes[
                signal_idx
            ].tolist()
        )

        cx: float = (x1 + x2) / 2.0
        cy: float = (y1 + y2) / 2.0

        candidates: tuple[int, ...] = ()

        if 0.0 <= cx < width and 0.0 <= cy < height:
            x_pixel: int = int(cx)
            y_pixel: int = int(cy)

            candidates = tuple(
                idx
                for idx in nucleus_indices
                if bool(masks[idx, y_pixel, x_pixel])
            )

        if len(candidates) == 1:
            status: AssociationStatus = "assigned"
            nucleus_idx: int | None = candidates[0]

            counts[nucleus_idx][class_idx] += 1

        elif len(candidates) > 1:
            status = "ambiguous"
            nucleus_idx = None

        else:
            status = "outside"
            nucleus_idx = None

        statuses[status] += 1

        assignments.append(
            SignalAssignment(
                signal_index=signal_idx,
                class_name=SIGNAL_NAMES[class_idx],
                confidence=confidence,
                box_xyxy=(x1, y1, x2, y2),
                center_xy=(cx, cy),
                status=status,
                nucleus_index=nucleus_idx,
                candidate_nuclei=candidates,
            )
        )

    nuclei: list[NucleusCount] = [
        NucleusCount(
            nucleus_index=idx,
            confidence=float(nucleus_scores[idx]),
            fusion=counts[idx][0],
            chr17=counts[idx][1],
            her2=counts[idx][2],
        )
        for idx in nucleus_indices
    ]

    return AssociationReport(
        nuclei=nuclei,
        signals=assignments,
        status_counts={
            key: int(statuses.get(key, 0))
            for key in (
                "assigned",
                "outside",
                "ambiguous",
            )
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count signals inside predicted nuclei"
    )

    parser.add_argument(
        "--weights", type=Path, required=True
    )
    parser.add_argument(
        "--image", type=Path, required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/association.json"),
    )
    parser.add_argument(
        "--signal-conf", type=float, default=0.25
    )
    parser.add_argument(
        "--nucleus-conf", type=float, default=0.50
    )
    parser.add_argument(
        "--nms-iou", type=float, default=0.70
    )
    parser.add_argument(
        "--device", type=str, default=None
    )

    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parse_args()

    with Image.open(args.image) as raw:
        image_array: np.ndarray = np.array(
            raw.convert("RGB"), copy=True
        )

    image: Tensor = (
        torch.from_numpy(image_array)
        .permute(2, 0, 1)
        .float()[None]
        / 255.0
    )

    predictor = DDishPredictor(
        weights=args.weights,
        device=args.device,
        conf=min(
            args.signal_conf,
            args.nucleus_conf,
        ),
        nms_iou=args.nms_iou,
    )

    prediction: PatchPredictions = (
        predictor.predict(image)[0]
    )

    report: AssociationReport = associate_signals(
        prediction,
        signal_conf=args.signal_conf,
        nucleus_conf=args.nucleus_conf,
    )

    args.output.parent.mkdir(
        parents=True, exist_ok=True
    )

    args.output.write_text(
        json.dumps(asdict(report), indent=2)
    )

    print(f"Nuclei retained: {len(report.nuclei)}")
    print(f"Signal status counts: {report.status_counts}")
    print("Nucleus | Fusion | Chr17 | HER2")

    for nucleus in report.nuclei:
        print(
            f"{nucleus.nucleus_index:>7} | "
            f"{nucleus.fusion:>6} | "
            f"{nucleus.chr17:>5} | "
            f"{nucleus.her2:>4}"
        )

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
