"""Load a custom D-DISH checkpoint and run inference."""

from pathlib import Path
from typing import Any

import torch

from src.inference.postprocess import (
    PatchPredictions,
    decode_batch,
)
from src.model import DDishYOLO


class DDishPredictor:

    def __init__(
        self,
        weights: str | Path,
        device: str | None = None,
        conf: float = 0.001,
        nms_iou: float = 0.7,
        max_det: int = 300,
    ) -> None:

        self.device = torch.device(
            device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        self.conf = conf
        self.nms_iou = nms_iou
        self.max_det = max_det

        checkpoint: dict[str, Any] = torch.load(
            weights,
            map_location="cpu",
            weights_only=True,
        )

        self.metadata: dict[str, Any] = {
            key: value
            for key, value in checkpoint.items()
            if key not in {"model", "optimizer", "scheduler"}
        }

        model = DDishYOLO(
           variant=checkpoint.get("variant", "n"),
           p2_detection=bool(
            checkpoint.get("p2_detection", False)
            ),
        )

        model.load_state_dict(
            checkpoint["model"],
            strict=True,
        )

        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def predict(
        self,
        images: torch.Tensor,
    ) -> list[PatchPredictions]:
        """
        images:
            (B, 3, H, W)
            RGB float32 values in [0, 1].
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"Expected (B, 3, H, W), "
                f"got {tuple(images.shape)}"
            )

        images = images.to(
            self.device,
            non_blocking=True,
        )

        outputs = self.model(images)

        return decode_batch(
            outputs,
            image_shape=(
                images.shape[-2],
                images.shape[-1],
            ),
            conf=self.conf,
            nms_iou=self.nms_iou,
            max_det=self.max_det,
        )
