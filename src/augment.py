"""Colour-preserving rotations and flips for paired D-DISH targets."""

from typing import Any

import torch
from torch import Tensor


def transform_boxes(
    boxes: Tensor,
    *,
    quarter_turns: int,
    horizontal_flip: bool,
) -> Tensor:
    """
    Transform normalized (cx, cy, w, h) boxes
    on square images.
    """
    result: Tensor = boxes.clone()

    if horizontal_flip:
        result[:, 0] = 1.0 - result[:, 0]

    for _ in range(quarter_turns % 4):
        cx, cy, width, height = result.unbind(dim=1)

        result = torch.stack(
            (cy, 1.0 - cx, height, width),
            dim=1,
        )

    return result


def geometric_transform(
    sample: dict[str, Any],
    *,
    quarter_turns: int,
    horizontal_flip: bool,
) -> dict[str, Any]:
    """
    Apply identical geometry to image, nucleus masks,
    and both sets of bounding boxes.
    """
    image: Tensor = sample["image"]
    masks: Tensor = sample["nucleus_masks"]

    if image.shape[-2] != image.shape[-1]:
        raise ValueError(
            "90-degree augmentation requires square patches"
        )

    if horizontal_flip:
        image = torch.flip(image, dims=(-1,))
        masks = torch.flip(masks, dims=(-1,))

    k: int = quarter_turns % 4

    if k:
        image = torch.rot90(
            image, k=k, dims=(-2, -1)
        )
        masks = torch.rot90(
            masks, k=k, dims=(-2, -1)
        )

    transformed: dict[str, Any] = sample.copy()

    transformed["image"] = image.contiguous()
    transformed["nucleus_masks"] = masks.contiguous()

    transformed["nucleus_boxes"] = transform_boxes(
        sample["nucleus_boxes"],
        quarter_turns=k,
        horizontal_flip=horizontal_flip,
    )

    transformed["signal_boxes"] = transform_boxes(
        sample["signal_boxes"],
        quarter_turns=k,
        horizontal_flip=horizontal_flip,
    )

    return transformed


class GeometricAugment:
    """
    Randomly select one of eight possible orientations
    for every training sample.
    """

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        k: int = int(
            torch.randint(0, 4, ()).item()
        )

        flip: bool = bool(
            torch.randint(0, 2, ()).item()
        )

        return geometric_transform(
            sample,
            quarter_turns=k,
            horizontal_flip=flip,
        )
