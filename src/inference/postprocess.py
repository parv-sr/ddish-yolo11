"""Decode the two YOLO11 heads into boxes and nucleus masks."""

from dataclasses import dataclass
from typing import Mapping

import torch
from ultralytics.utils.ops import process_mask

try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression


@dataclass
class Detections:
    boxes: torch.Tensor
    scores: torch.Tensor
    classes: torch.Tensor
    masks: torch.Tensor | None = None


@dataclass
class PatchPredictions:
    nuclei: Detections
    signals: Detections


def decode_batch(
    outputs: Mapping[str, object],
    image_shape: tuple[int, int],
    conf: float = 0.001,
    nms_iou: float = 0.7,
    max_det: int = 300,
) -> list[PatchPredictions]:
    """
    Decode evaluation-mode model outputs.

    Boxes: xyxy pixel coordinates.
    Masks: individual binary nucleus masks at input resolution.
    """
    seg_output = outputs["nuclei"]
    det_output = outputs["signals"]

    # Segment.eval() -> ((decoded, prototypes), raw)
    # Detect.eval()  -> (decoded, raw)

    if (
        not isinstance(seg_output, tuple)
        or not isinstance(seg_output[0], tuple)
    ):
        raise TypeError(
            "Expected Segment evaluation output. Call model.eval()."
        )

    if not isinstance(det_output, tuple):
        raise TypeError(
            "Expected Detect evaluation output. Call model.eval()."
        )

    seg_decoded, prototypes = seg_output[0]
    det_decoded = det_output[0]

    if not all(
        torch.is_tensor(t)
        for t in (seg_decoded, prototypes, det_decoded)
    ):
        raise TypeError("Unexpected YOLO head output types")

    nucleus_rows = non_max_suppression(
        seg_decoded,
        conf_thres=conf,
        iou_thres=nms_iou,
        nc=1,
        max_det=max_det,
        agnostic=False,
    )

    signal_rows = non_max_suppression(
        det_decoded,
        conf_thres=conf,
        iou_thres=nms_iou,
        nc=3,
        max_det=max_det,
        agnostic=False,
    )

    decoded: list[PatchPredictions] = []

    for index, (nucleus, signal) in enumerate(
        zip(nucleus_rows, signal_rows)
    ):
        # Nucleus rows:
        # [x1, y1, x2, y2, confidence, class, mask_coefficients...]

        if len(nucleus) == 0:
            nucleus_masks = torch.zeros(
                (0, *image_shape),
                dtype=torch.bool,
                device=prototypes.device,
            )
        else:
            nucleus_masks = process_mask(
                prototypes[index],
                nucleus[:, 6:],
                nucleus[:, :4],
                image_shape,
                upsample=True,
            ).bool()

        decoded.append(
            PatchPredictions(
                nuclei=Detections(
                    boxes=nucleus[:, :4],
                    scores=nucleus[:, 4],
                    classes=nucleus[:, 5].long(),
                    masks=nucleus_masks,
                ),
                signals=Detections(
                    boxes=signal[:, :4],
                    scores=signal[:, 4],
                    classes=signal[:, 5].long(),
                ),
            )
        )

    return decoded
