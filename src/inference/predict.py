"""CLI for visualizing predicted nuclei and gene signals."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.inference.predictor import DDishPredictor


NUCLEUS_COLORS: tuple[tuple[int, int, int], ...] = (
    (53, 153, 219),
    (54, 190, 139),
    (195, 126, 216),
    (251, 161, 72),
)

SIGNAL_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 202, 40),  # Fusion: yellow
    (245, 75, 158),  # Chr17: pink
    (60, 220, 85),   # HER2: green
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict nuclei and signals in one patch"
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
        default=Path("runs/prediction.png"),
    )

    parser.add_argument(
        "--conf", type=float, default=0.25
    )

    parser.add_argument(
        "--nms-iou", type=float, default=0.7
    )

    parser.add_argument(
        "--device", type=str, default=None
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with Image.open(args.image) as source:
        original = source.convert("RGB")

    pixels = np.asarray(
        original,
        dtype=np.uint8,
    ).copy()

    image_tensor = (
        torch.from_numpy(pixels)
        .permute(2, 0, 1)
        .float()[None]
        / 255.0
    )

    predictor = DDishPredictor(
        args.weights,
        device=args.device,
        conf=args.conf,
        nms_iou=args.nms_iou,
    )

    output = predictor.predict(image_tensor)[0]

    layer = Image.new(
        "RGBA",
        original.size,
        (0, 0, 0, 0),
    )

    masks = output.nuclei.masks

    if masks is None:
        raise RuntimeError("Missing predicted nucleus masks")

    for index, mask in enumerate(
        masks.detach().cpu().numpy()
    ):
        alpha = Image.fromarray(
            mask.astype(np.uint8) * 255
        )

        color = NUCLEUS_COLORS[
            index % len(NUCLEUS_COLORS)
        ]

        layer.paste(
            (*color, 85),
            (0, 0),
            alpha,
        )

    painted = Image.alpha_composite(
        original.convert("RGBA"),
        layer,
    ).convert("RGB")

    canvas = ImageDraw.Draw(painted)

    for box, class_id in zip(
        output.signals.boxes,
        output.signals.classes,
    ):
        color = SIGNAL_COLORS[int(class_id.item())]

        canvas.rectangle(
            tuple(float(v) for v in box.tolist()),
            outline=color,
            width=2,
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    painted.save(args.output)

    print(f"Saved {args.output}")
    print(
        f"Nuclei: {len(output.nuclei.boxes)}, "
        f"signals: {len(output.signals.boxes)}"
    )
    print(
        "Signal colors: Fusion=yellow, "
        "Chr17=pink, HER2=green"
    )


if __name__ == "__main__":
    main()
