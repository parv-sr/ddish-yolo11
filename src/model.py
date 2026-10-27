import torch
from torch import nn

from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect, Segment


def transfer_compatible_weights(source, destination):
    """
    Transfer pretrained parameters whose names and shapes match.

    Classification layers with incompatible shapes remain initialized
    for our new tasks.
    """
    source_state = source.state_dict()
    target_state = destination.state_dict()

    compatible = {
        name: weight
        for name, weight in source_state.items()
        if name in target_state
        and weight.shape == target_state[name].shape
    }

    destination.load_state_dict(compatible, strict=False)

    return len(compatible), len(target_state)


def initialize_head(head, pretrained_head):
    """Initialize strides, biases, and compatible pretrained weights."""

    # Register strides as a buffer so model.to("cuda") moves them too.
    strides = pretrained_head.stride.detach().clone()

    del head.stride
    head.register_buffer("stride", strides)

    head.bias_init()

    return transfer_compatible_weights(pretrained_head, head)


class DDishYOLO(nn.Module):
    """
    Unified YOLO11 model with:
      - Shared pretrained backbone + neck
      - One-class nucleus instance-segmentation head
      - Three-class gene-signal detection head
    """

    def __init__(self, variant="n"):
        super().__init__()

        # Load pretrained YOLO11 checkpoints.
        seg_model = YOLO(f"yolo11{variant}-seg.pt").model
        det_model = YOLO(f"yolo11{variant}.pt").model

        seg_pretrained = seg_model.model[-1]
        det_pretrained = det_model.model[-1]

        if not isinstance(seg_pretrained, Segment):
            raise TypeError("Expected a YOLO11 Segment head")

        if not isinstance(det_pretrained, Detect):
            raise TypeError("Expected a YOLO11 Detect head")

        # All layers except the original final segmentation head.
        # This includes the backbone and multiscale feature neck.
        self.shared = seg_model.model[:-1]
        self.shared.requires_grad_(True)    # Unfreeze base model weights for finetuning

        # Ultralytics layers reference earlier outputs using .f
        # and identify themselves using .i.
        self.feature_indices = tuple(seg_pretrained.f)

        self.save = set(seg_model.save) | set(self.feature_indices)

        # Determine the P3/P4/P5 channel widths from the
        # original pretrained segmentation head.
        channels = tuple(
            branch[0].conv.in_channels
            for branch in seg_pretrained.cv2
        )

        det_channels = tuple(
            branch[0].conv.in_channels
            for branch in det_pretrained.cv2
        )

        if channels != det_channels:
            raise ValueError(
                "Detection and segmentation feature channels differ"
            )

        # One-class instance segmentation.
        self.nucleus_head = Segment(
            nc=1,
            nm=seg_pretrained.nm,
            npr=seg_pretrained.npr,
            reg_max=seg_pretrained.reg_max,
            ch=channels,
        )

        # Three-class gene-signal detection.
        self.signal_head = Detect(
            nc=3,
            reg_max=det_pretrained.reg_max,
            ch=channels,
        )

        seg_copied, seg_total = initialize_head(
            self.nucleus_head, seg_pretrained
        )

        det_copied, det_total = initialize_head(
            self.signal_head, det_pretrained
        )

        print(f"Feature layers: {self.feature_indices}")
        print(f"Feature channels: {channels}")
        print(f"Segmentation head: {seg_copied}/{seg_total} weights transferred")
        print(f"Detection head: {det_copied}/{det_total} weights transferred")

        self.train()

    def extract_features(self, x):
        """
        Execute YOLO11's shared backbone and neck,
        respecting its skip connections.
        """
        cached = [None] * len(self.shared)

        for layer in self.shared:

            if layer.f == -1:
                layer_input = x

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

        features = [cached[i] for i in self.feature_indices]

        if any(feature is None for feature in features):
            raise RuntimeError("Missing a required feature map")

        return features

    def forward(self, images):
        features = self.extract_features(images)

        # Both heads receive the SAME shared features.
        nucleus_predictions = self.nucleus_head(features.copy())
        signal_predictions = self.signal_head(features.copy())

        return {
            "nuclei": nucleus_predictions,
            "signals": signal_predictions,
        }
