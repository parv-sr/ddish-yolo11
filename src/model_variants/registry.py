"""Single model factory for training, inference, and diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from src.model import DDishYOLO
from src.model_variants.classifier_transfer import ClassifierTransferYOLO
from src.model_variants.p2_segmentation import P2SegmentationYOLO

MODEL_REGISTRY: dict[str, type[DDishYOLO]] = {
    "baseline": DDishYOLO,
    "classifier_transfer": ClassifierTransferYOLO,
    "p2_segmentation": P2SegmentationYOLO,
}
MODEL_NAMES: tuple[str, ...] = tuple(MODEL_REGISTRY)
MODEL_SIZES: tuple[str, ...] = ("n", "s")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str = "baseline"
    size: str = "n"
    p2_detection: bool = False

    def __post_init__(self) -> None:
        if self.name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model variant: {self.name}")

        if self.size not in MODEL_SIZES:
            raise ValueError(f"Unsupported YOLO11 size: {self.size}")

        if self.name == "p2_segmentation" and not self.p2_detection:
            raise ValueError("P2 segmentation requires P2 detection")

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> ModelConfig:
        stored: Any = checkpoint.get("model_config")

        if stored is None:
            return cls(size=str(checkpoint.get("variant", "n")), p2_detection=bool(checkpoint.get("p2_detection", False)))

        if not isinstance(stored, Mapping):
            raise TypeError("Checkpoint model_config must be a mapping")

        return cls(
            name=str(stored.get("name", "baseline")),
            size=str(stored.get("size", checkpoint.get("variant", "n"))),
            p2_detection=bool(stored.get("p2_detection", checkpoint.get("p2_detection", False))),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_model(config: ModelConfig) -> DDishYOLO:
    model_type: type[DDishYOLO] = MODEL_REGISTRY[config.name]
    return model_type(variant=config.size, p2_detection=config.p2_detection)
