"""High-resolution P2 nucleus segmentation based on classifier-transfer YOLO."""

from copy import deepcopy
from typing import Any

from torch import Tensor, nn
from ultralytics.nn.modules.head import Segment

from src.model_variants.classifier_transfer import ClassifierTransferYOLO


def prepend_p2_segment_head(old: Segment) -> Segment:
    channels: tuple[int, ...] = tuple(branch[0].conv.in_channels for branch in old.cv2)
    head: Segment = Segment(nc=old.nc, nm=old.nm, npr=old.npr, reg_max=old.reg_max, ch=(channels[0], *channels))

    del head.stride
    head.register_buffer("stride", old.stride.new_tensor([4.0, *old.stride.tolist()]))
    head.bias_init()

    for name in ("cv2", "cv3", "cv4"):
        source: nn.ModuleList = getattr(old, name)
        target: nn.ModuleList = getattr(head, name)

        for index, branch in enumerate(source):
            target[index + 1] = deepcopy(branch)

    head.proto = deepcopy(old.proto)
    head.dfl = deepcopy(old.dfl)
    return head


class P2SegmentationYOLO(ClassifierTransferYOLO):
    def __init__(self, variant: str = "n", p2_detection: bool = True) -> None:
        if not p2_detection:
            raise ValueError("P2 segmentation requires P2 detection")

        super().__init__(variant=variant, p2_detection=p2_detection)
        self.nucleus_head = prepend_p2_segment_head(self.nucleus_head)

    def forward(self, images: Tensor) -> dict[str, Any]:
        _, signal_features = self.extract_features(images)
        return {
            "nuclei": self.nucleus_head(signal_features.copy()),
            "signals": self.signal_head(signal_features.copy()),
        }
