import torch
from torch.utils.data import DataLoader

from src.dataset import (
    DDishDataset,
    collate_ddish,
    load_records,
    split_by_source,
)
from src.model import DDishYOLO


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    records = load_records("data/raw")
    train, val, _ = split_by_source(records, seed=42)

    loader = DataLoader(
        DDishDataset(train),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_ddish,
    )

    batch = next(iter(loader))
    images = batch["images"].to(device)

    model = DDishYOLO(variant="n").to(device)
    model.train()

    print("\nInput:", tuple(images.shape))

    predictions = model(images)

    for task_name, output in predictions.items():
        print(f"\n{task_name.upper()} HEAD")

        if not isinstance(output, dict):
            raise TypeError(
                f"Expected raw training predictions as a dict, "
                f"got {type(output).__name__}"
            )

        for name, value in output.items():
            if isinstance(value, torch.Tensor):
                print(f"  {name}: {tuple(value.shape)}")

            elif isinstance(value, (list, tuple)):
                print(
                    f"  {name}: "
                    f"{[tuple(t.shape) for t in value]}"
                )

    # Check the essential task-specific outputs.
    nuclei = predictions["nuclei"]
    signals = predictions["signals"]

    assert nuclei["scores"].shape[1] == 1
    assert signals["scores"].shape[1] == 3
    assert "proto" in nuclei

    # Diagnostic gradient test only, NOT the training loss.
    probe = (
        nuclei["scores"].square().mean()
        + nuclei["proto"].square().mean()
        + signals["scores"].square().mean()
    )

    probe.backward()

    for name, module in [
        ("Shared network", model.shared),
        ("Nucleus head", model.nucleus_head),
        ("Signal head", model.signal_head),
    ]:
        has_gradient = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in module.parameters()
        )

        assert has_gradient, f"No valid gradient in {name}"
        print(f"{name}: gradients OK")

    print("\nMODEL SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
