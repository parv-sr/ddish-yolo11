from typing import SimpleNamespace

import torch

from ultralytics.utils.loss import (
    v8DetectionLoss,
    v8SegmentationLoss,
)

class HeadAdapter:
    """
    Exposes one of our task heads in the format expected
    by Ultralytics' loss constructors.
    """

    def __init__(self, head):
        self.model = [head]

        self.args = SimpleNamespace(
            box=7.5,
            cls=0.5,
            dfl=1.5,
            overlap_mask=False,
        )

    def parameters(self):
        return self.model[-1].parameters()


def prepare_targets(targets, task, device):
    """
    Convert our list of per-image targets into the flattened
    batch format expected by Ultralytics.

    Critical: the ordering of nucleus boxes and masks
    must remain identical.
    """

    batch_indices = []
    classes = []
    boxes = []
    masks = []

    for image_index, target in enumerate(targets):

        if task == "nuclei":
            current_boxes = target["nucleus_boxes"].to(device)
            current_classes = torch.zeros(
                (len(current_boxes), 1),
                dtype=torch.float32,
                device=device,
            )

            current_masks = target["nucleus_masks"].to(
                device=device,
                dtype=torch.float32,
            )

            assert len(current_boxes) == len(current_masks)
            masks.append(current_masks)

        elif task == "signals":
            current_boxes = target["signal_boxes"].to(device)
            current_classes = (
                target["signal_classes"]
                .to(device)
                .float()
                .reshape(-1, 1)
            )

        else:
            raise ValueError(f"Unknown task: {task}")

        batch_indices.append(
            torch.full(
                (len(current_boxes),),
                image_index,
                dtype=torch.long,
                device=device,
            )
        )

        classes.append(current_classes)
        boxes.append(current_boxes)

    result = {
        "batch_idx": torch.cat(batch_indices),
        "cls": torch.cat(classes),
        "bboxes": torch.cat(boxes),
    }

    if task == "nuclei":
        result["masks"] = torch.cat(masks)

    return result


class DDishLoss:
    """
    Combined nucleus segmentation + gene-signal detection loss.
    """

    def __init__(self, model, lambda_seg=1.0, lambda_det=1.0):

        self.seg_criterion = v8SegmentationLoss(
            HeadAdapter(model.nucleus_head)
        )

        self.det_criterion = v8DetectionLoss(
            HeadAdapter(model.signal_head)
        )

        self.lambda_seg = lambda_seg
        self.lambda_det = lambda_det

    def __call__(self, predictions, targets):

        device = predictions["signals"]["scores"].device

        nucleus_targets = prepare_targets(
            targets, "nuclei", device
        )

        signal_targets = prepare_targets(
            targets, "signals", device
        )

        seg_losses, seg_components = self.seg_criterion(
            predictions["nuclei"], nucleus_targets
        )

        det_losses, det_components = self.det_criterion(
            predictions["signals"], signal_targets
        )

        # Ultralytics returns component vectors,
        # not a single scalar loss.
        seg_loss = seg_losses.sum()
        det_loss = det_losses.sum()

        total = (
            self.lambda_seg * seg_loss
            + self.lambda_det * det_loss
        )

        details = {
            "segmentation": seg_loss.detach(),
            "detection": det_loss.detach(),
            "seg_components": seg_components,
            "det_components": det_components,
        }

        return total, details
