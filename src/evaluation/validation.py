"""Shared validation split and one-to-one matching helpers."""

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from src.dataset import load_records, split_by_source, split_random


def validation_records(metadata: Mapping[str, Any], root: Path = Path("data/raw"), seed: int = 42) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = load_records(root)
    strategy: str = str(metadata.get("split", "source"))

    if strategy == "random":
        _, validation, identifiers = split_random(records, seed=seed, val_fraction=0.2)
    elif strategy == "source":
        _, validation, identifiers = split_by_source(records, seed=seed)
    else:
        raise ValueError(f"Unsupported validation split: {strategy}")

    saved: Any = metadata.get("split_ids", metadata.get("held_out"))

    if saved is None or saved != identifiers:
        raise ValueError("Validation identities differ from checkpoint; check data root and seed")

    return validation


def greedy_matches(ious: Tensor, scores: Tensor, threshold: float = 0.5) -> dict[int, int]:
    """Match predictions to unique targets by descending confidence."""
    if ious.ndim != 2 or ious.shape[0] != len(scores) or not 0 < threshold <= 1:
        raise ValueError("Invalid IoU matrix, score vector, or threshold")

    matched: dict[int, int] = {}
    claimed: set[int] = set()

    if ious.shape[1] == 0:
        return matched

    for prediction_index in scores.argsort(descending=True, stable=True).tolist():
        row: Tensor = ious[prediction_index].clone()

        if claimed:
            row[list(claimed)] = -1

        overlap, target_index = row.max(dim=0)

        if float(overlap) >= threshold:
            matched[int(prediction_index)] = int(target_index)
            claimed.add(int(target_index))

    return matched
