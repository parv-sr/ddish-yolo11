"""YOLO11 classifier-head transfer that preserves pretrained hidden layers."""

from copy import deepcopy

import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

from src.model import DDishYOLO


def transfer_classifier_trunks(destination: Detect, source: Detect) -> int:
    """Copy corresponding pretrained classifiers, except their final COCO outputs."""
    source_levels: dict[int, int] = {int(stride): i for i, stride in enumerate(source.stride.tolist())}
    copied: int = 0

    for target_index, stride in enumerate(destination.stride.tolist()):
        source_index: int | None = source_levels.get(int(stride))

        if source_index is None:
            continue

        input_channels: int = source.cv2[source_index][0].conv.in_channels
        target_channels: int = destination.cv2[target_index][0].conv.in_channels

        if input_channels != target_channels:
            raise ValueError(f"Feature channels differ at stride {stride}")

        branch: nn.Sequential = deepcopy(source.cv3[source_index])
        previous: nn.Module = branch[-1]
        current: nn.Module = destination.cv3[target_index][-1]

        if not isinstance(previous, nn.Conv2d) or not isinstance(current, nn.Conv2d):
            raise TypeError("Unexpected classifier output type")

        if previous.kernel_size != (1, 1) or current.bias is None:
            raise ValueError("Expected a biased 1x1 classifier output")

        replacement: nn.Conv2d = nn.Conv2d(previous.in_channels, destination.nc, kernel_size=1)
        replacement = replacement.to(device=previous.weight.device, dtype=previous.weight.dtype)

        with torch.no_grad():
            replacement.bias.copy_(current.bias)

        branch[-1] = replacement
        branch.requires_grad_(True)
        branch.train(destination.training)
        destination.cv3[target_index] = branch
        copied += sum(parameter.numel() for parameter in branch[:-1].parameters())

    return copied


class ClassifierTransferYOLO(DDishYOLO):
    def __init__(self, variant: str = "n", p2_detection: bool = False) -> None:
        super().__init__(variant=variant, p2_detection=p2_detection)

        pretrained: Detect = YOLO(f"yolo11{variant}.pt").model.model[-1]
        copied: int = transfer_classifier_trunks(self.signal_head, pretrained)
        print(f"Preserved pretrained classifier-trunk parameters: {copied:,}")
