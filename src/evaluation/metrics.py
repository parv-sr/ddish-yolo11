"""Precision, recall, F1 and interpolated average precision."""

from collections.abc import Sequence

import numpy as np
import torch

from ultralytics.utils.metrics import box_iou


IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)


def normalized_boxes_to_xyxy(
    boxes: torch.Tensor,
    image_shape: tuple[int, int],
) -> torch.Tensor:
    """Normalized xywh -> pixel xyxy."""
    height, width = image_shape

    scaled = boxes * boxes.new_tensor([
        width, height, width, height
    ])

    centers = scaled[:, :2]
    sizes = scaled[:, 2:]

    return torch.cat(
        (centers - sizes / 2, centers + sizes / 2),
        dim=1,
    )


def pairwise_box_iou(
    pred: torch.Tensor,
    true: torch.Tensor,
) -> torch.Tensor:
    if len(pred) == 0 or len(true) == 0:
        return pred.new_zeros((len(pred), len(true)))

    return box_iou(pred, true)


def pairwise_mask_iou(
    pred: torch.Tensor,
    true: torch.Tensor,
) -> torch.Tensor:
    """Compute exact pixel-wise IoU for all mask pairs."""
    if len(pred) == 0 or len(true) == 0:
        return pred.new_zeros(
            (len(pred), len(true)),
            dtype=torch.float32,
        )

    if pred.shape[1:] != true.shape[1:]:
        raise ValueError(
            "Prediction and GT mask resolutions differ"
        )

    p = pred.reshape(len(pred), -1).float()
    t = true.reshape(len(true), -1).float()

    intersections = p @ t.T

    unions = (
        p.sum(1)[:, None]
        + t.sum(1)[None, :]
        - intersections
    )

    return intersections / unions.clamp(min=1)


def interpolated_ap(
    tp: np.ndarray,
    n_gt: int,
) -> float:
    """101-point interpolated AP at one IoU threshold."""
    if n_gt == 0 or len(tp) == 0 or not tp.any():
        return 0.0

    recall = np.cumsum(tp) / n_gt

    precision = (
        np.cumsum(tp)
        / np.arange(1, len(tp) + 1)
    )

    recall = np.concatenate((
        [0.0], recall, [1.0]
    ))

    precision = np.concatenate((
        [0.0], precision, [0.0]
    ))

    precision = np.maximum.accumulate(
        precision[::-1]
    )[::-1]

    points = np.linspace(0.0, 1.0, 101)

    positions = np.searchsorted(
        recall,
        points,
        side="left",
    )

    return float(precision[positions].mean())


class APAccumulator:
    """Accumulate confidence-ranked predictions across images."""

    def __init__(
        self,
        class_names: Sequence[str],
        fixed_conf: float = 0.25,
    ) -> None:

        self.class_names = tuple(class_names)
        self.fixed_conf = fixed_conf

        self.gt_count = np.zeros(
            len(self.class_names),
            dtype=np.int64,
        )

        self.scores: list[list[np.ndarray]] = [
            [] for _ in self.class_names
        ]

        self.matches: list[list[np.ndarray]] = [
            [] for _ in self.class_names
        ]

        self.matched_dice: list[float] = []

    def update(
        self,
        scores: torch.Tensor,
        pred_classes: torch.Tensor,
        gt_classes: torch.Tensor,
        ious: torch.Tensor,
        track_dice: bool = False,
    ) -> None:
        """
        Update using one image.

        ious has shape (num_predictions, num_ground_truth).
        """
        conf = (
            scores.detach().float().cpu().numpy()
        )

        predicted = (
            pred_classes.detach().long().cpu().numpy()
        )

        labels = (
            gt_classes.detach().long().cpu().numpy()
        )

        overlaps = (
            ious.detach().float().cpu().numpy()
        )

        if overlaps.shape != (len(conf), len(labels)):
            raise ValueError("IoU matrix shape mismatch")

        for cls in range(len(self.class_names)):

            pred_ids = np.flatnonzero(
                predicted == cls
            )

            pred_ids = pred_ids[
                np.argsort(
                    -conf[pred_ids],
                    kind="stable",
                )
            ]

            gt_ids = np.flatnonzero(labels == cls)

            self.gt_count[cls] += len(gt_ids)

            correct = np.zeros(
                (len(pred_ids), len(IOU_THRESHOLDS)),
                dtype=bool,
            )

            class_ious = overlaps[
                np.ix_(pred_ids, gt_ids)
            ]

            for threshold_index, threshold in enumerate(
                IOU_THRESHOLDS
            ):
                claimed = np.zeros(
                    len(gt_ids),
                    dtype=bool,
                )

                for row_index, row in enumerate(class_ious):

                    available = np.flatnonzero(
                        (row >= threshold) & ~claimed
                    )

                    if len(available) == 0:
                        continue

                    match = available[
                        np.argmax(row[available])
                    ]

                    claimed[match] = True
                    correct[
                        row_index, threshold_index
                    ] = True

                    if (
                        track_dice
                        and threshold_index == 0
                        and conf[pred_ids[row_index]]
                        >= self.fixed_conf
                    ):
                        overlap = float(row[match])
                        self.matched_dice.append(
                            2 * overlap / (1 + overlap)
                        )

            self.scores[cls].append(
                conf[pred_ids]
            )

            self.matches[cls].append(correct)

    def summarize(self) -> dict[str, object]:
        rows: list[dict[str, object]] = []

        for cls, name in enumerate(self.class_names):

            conf = np.concatenate(
                self.scores[cls]
            )

            correct = np.concatenate(
                self.matches[cls],
                axis=0,
            )

            order = np.argsort(
                -conf,
                kind="stable",
            )

            conf = conf[order]
            correct = correct[order]

            n_gt = int(self.gt_count[cls])

            selected = conf >= self.fixed_conf

            tp = int(correct[selected, 0].sum())
            fp = int(selected.sum()) - tp
            fn = n_gt - tp

            precision = (
                tp / (tp + fp)
                if tp + fp else 0.0
            )

            recall = (
                tp / n_gt
                if n_gt else 0.0
            )

            f1 = (
                2 * precision * recall
                / (precision + recall)
                if precision + recall else 0.0
            )

            aps = [
                interpolated_ap(
                    correct[:, index],
                    n_gt,
                )
                for index in range(len(IOU_THRESHOLDS))
            ]

            rows.append({
                "class": name,
                "gt": n_gt,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ap50": aps[0] if n_gt else None,
                "map50_95": (
                    float(np.mean(aps))
                    if n_gt else None
                ),
            })

        evaluable = [
            row for row in rows if row["gt"] > 0
        ]

        summary: dict[str, object] = {
            "classes": rows,
            "macro": {
                key: (
                    float(np.mean([
                        float(row[key])
                        for row in evaluable
                    ]))
                    if evaluable else None
                )
                for key in (
                    "precision",
                    "recall",
                    "f1",
                    "ap50",
                    "map50_95",
                )
            },
        }

        if self.matched_dice:
            summary[
                "matched_instance_dice_at_iou50"
            ] = float(np.mean(self.matched_dice))

            summary[
                "matched_instance_count"
            ] = len(self.matched_dice)

        return summary
