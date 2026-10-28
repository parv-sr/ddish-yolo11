from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ultralytics import YOLO
from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.head import Detect, Segment


def transfer_compatible_weights(
    source: nn.Module,
    destination: nn.Module,
    level_shift: int = 0,
) -> tuple[int, int]:
    """
    Transfer compatible pretrained weights.

    If level_shift=1, remap original detection branches:
        P3: 0 -> 1
        P4: 1 -> 2
        P5: 2 -> 3

    The new P2 branch (index 0) remains freshly initialized.
    """
    source_state = source.state_dict()
    destination_state = destination.state_dict()

    compatible: dict[str, Tensor] = {}

    for name, weight in source_state.items():
        target_name = name

        parts = name.split(".", 2)

        if (
            level_shift
            and len(parts) == 3
            and parts[0] in {"cv2", "cv3"}
            and parts[1].isdigit()
        ):
            target_name = (
                f"{parts[0]}."
                f"{int(parts[1]) + level_shift}."
                f"{parts[2]}"
            )

        if (
            target_name in destination_state
            and weight.shape == destination_state[target_name].shape
        ):
            compatible[target_name] = weight

    destination.load_state_dict(
        compatible,
        strict=False,
    )

    return len(compatible), len(destination_state)


def initialize_head(
    head: Detect,
    pretrained_head: Detect,
    strides: tuple[float, ...] | None = None,
    level_shift: int = 0,
) -> tuple[int, int]:
    """Initialize a task head and copy compatible weights."""

    if strides is None:
        stride_tensor = (
            pretrained_head.stride.detach().clone()
        )
    else:
        stride_tensor = torch.tensor(
            strides,
            dtype=pretrained_head.stride.dtype,
            device=pretrained_head.stride.device,
        )

    if len(stride_tensor) != head.nl:
        raise ValueError(
            "Stride count must match number of head levels"
        )

    del head.stride
    head.register_buffer("stride", stride_tensor)

    head.bias_init()

    return transfer_compatible_weights(
        pretrained_head,
        head,
        level_shift=level_shift,
    )


class DDishYOLO(nn.Module):
    """
    Unified YOLO11 model.

    Segmentation always uses P3/P4/P5.

    Detection uses:
      - P3/P4/P5 when p2_detection=False
      - P2/P3/P4/P5 when p2_detection=True
    """

    def __init__(
        self,
        variant: str = "n",
        p2_detection: bool = False,
    ) -> None:
        super().__init__()

        self.variant = variant
        self.p2_detection = p2_detection

        seg_model = YOLO(
            f"yolo11{variant}-seg.pt"
        ).model

        det_model = YOLO(
            f"yolo11{variant}.pt"
        ).model

        seg_pretrained = seg_model.model[-1]
        det_pretrained = det_model.model[-1]

        if not isinstance(seg_pretrained, Segment):
            raise TypeError("Expected a Segment head")

        if not isinstance(det_pretrained, Detect):
            raise TypeError("Expected a Detect head")

        self.shared = seg_model.model[:-1]

        # Pretrained checkpoint parameters may be frozen.
        self.shared.requires_grad_(True)

        self.feature_indices: tuple[int, ...] = tuple(
            int(index)
            for index in seg_pretrained.f
        )

        # YOLO11 backbone layer 2 is the P2/4 feature.
        self.p2_index = 2

        self.save: set[int] = (
            set(seg_model.save)
            | set(self.feature_indices)
        )

        if self.p2_detection:
            self.save.add(self.p2_index)

        # P3/P4/P5 channels.
        channels: tuple[int, ...] = tuple(
            branch[0].conv.in_channels
            for branch in seg_pretrained.cv2
        )

        det_channels: tuple[int, ...] = tuple(
            branch[0].conv.in_channels
            for branch in det_pretrained.cv2
        )

        if channels != det_channels:
            raise ValueError(
                "Pretrained feature channels do not match"
            )

        # Keep this head identical to Experiment 02.
        self.nucleus_head = Segment(
            nc=1,
            nm=seg_pretrained.nm,
            npr=seg_pretrained.npr,
            reg_max=seg_pretrained.reg_max,
            ch=channels,
        )

        seg_copied, seg_total = initialize_head(
            self.nucleus_head,
            seg_pretrained,
        )

        # Add the P2 fusion module only when enabled.
        if self.p2_detection:

            p2_channels: int = int(
                self.shared[
                    self.p2_index
                ].cv2.conv.out_channels
            )

            p3_channels: int = channels[0]

            # Concatenate upsampled P3 with backbone P2.
            # Refine the combined features using C3k2.
            self.p2_fusion: nn.Module = C3k2(
                c1=p3_channels + p2_channels,
                c2=p3_channels,
                n=1,
                c3k=False,
            )

            signal_channels = (
                p3_channels,
                *channels,
            )

            signal_strides = (
                4.0, 8.0, 16.0, 32.0
            )

            level_shift = 1

        else:
            self.p2_fusion = nn.Identity()

            signal_channels = channels
            signal_strides = (8.0, 16.0, 32.0)
            level_shift = 0

        self.signal_head = Detect(
            nc=3,
            reg_max=det_pretrained.reg_max,
            ch=signal_channels,
        )

        det_copied, det_total = initialize_head(
            self.signal_head,
            det_pretrained,
            strides=signal_strides,
            level_shift=level_shift,
        )

        print(f"P2 detection: {self.p2_detection}")
        print(f"Feature layers: {self.feature_indices}")
        print(f"Feature channels: {channels}")
        print(f"Signal strides: {signal_strides}")

        print(
            "Segmentation head: "
            f"{seg_copied}/{seg_total} weights transferred"
        )

        print(
            "Detection head: "
            f"{det_copied}/{det_total} weights transferred"
        )

        self.train()

    def extract_features(
        self,
        images: Tensor,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """
        Extract features from the shared YOLO11 backbone/neck.

        Returns:
            nucleus_features: P3/P4/P5
            signal_features: P3/P4/P5 or P2/P3/P4/P5
        """

        x: Tensor = images

        cached: list[Tensor | None] = [
            None
        ] * len(self.shared)

        for layer in self.shared:

            if layer.f == -1:
                layer_input: Any = x

            elif isinstance(layer.f, int):
                layer_input = cached[layer.f]

            else:
                layer_input = [
                    x if index == -1 else cached[index]
                    for index in layer.f
                ]

            x = layer(layer_input)

            if layer.i in self.save:
                cached[layer.i] = x

        nucleus_features: list[Tensor] = []

        for index in self.feature_indices:
            feature = cached[index]

            if feature is None:
                raise RuntimeError(
                    f"Missing feature from layer {index}"
                )

            nucleus_features.append(feature)

        signal_features = nucleus_features.copy()

        if self.p2_detection:

            p2_backbone = cached[self.p2_index]

            if p2_backbone is None:
                raise RuntimeError(
                    "Missing backbone P2 feature"
                )

            p3 = nucleus_features[0]

            upsampled_p3 = F.interpolate(
                p3,
                size=p2_backbone.shape[-2:],
                mode="nearest",
            )

            fused = torch.cat(
                (upsampled_p3, p2_backbone),
                dim=1,
            )

            p2 = self.p2_fusion(fused)

            signal_features = [
                p2,
                *nucleus_features,
            ]

        return nucleus_features, signal_features

    def forward(
        self,
        images: Tensor,
    ) -> dict[str, Any]:

        nucleus_features, signal_features = (
            self.extract_features(images)
        )

        nucleus_predictions = self.nucleus_head(
            nucleus_features
        )

        signal_predictions = self.signal_head(
            signal_features
        )

        return {
            "nuclei": nucleus_predictions,
            "signals": signal_predictions,
        }
