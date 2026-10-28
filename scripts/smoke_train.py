import torch
from torch.utils.data import DataLoader

from src.dataset import (
    DDishDataset,
    collate_ddish,
    load_records,
    split_by_source,
)
from src.model import DDishYOLO
from src.losses import DDishLoss


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    records = load_records("data/raw")
    train, _, _ = split_by_source(records, seed=42)

    loader = DataLoader(
        DDishDataset(train),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_ddish,
    )

    batch = next(iter(loader))

    images = batch["images"].to(device)
    targets = batch["targets"]

    model = DDishYOLO(variant="n", p2_detection=True).to(device)
    model.train()

    criterion = DDishLoss(model)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=1e-4,
        weight_decay=1e-4,
    )

    print("\nInput:", tuple(images.shape))

    optimizer.zero_grad(set_to_none=True)

    predictions = model(images)

    loss, details = criterion(predictions, targets)

    print("\nSegmentation loss:", details["segmentation"].item())
    print("Detection loss:", details["detection"].item())
    print("Total loss:", loss.item())

    print("\nSegmentation components:")
    for name, value in details["seg_components"].items():
        print(f"  {name}: {value.item():.6f}")

    print("\nDetection components:")
    for name, value in details["det_components"].items():
        print(f"  {name}: {value.item():.6f}")

    assert loss.ndim == 0
    assert torch.isfinite(loss), "Loss is NaN or Inf"

    loss.backward()

    for name, module in [
        ("Shared", model.shared),
        ("Nucleus", model.nucleus_head),
        ("Signals", model.signal_head),
    ]:
        gradients = [
            p.grad for p in module.parameters()
            if p.grad is not None
        ]

        assert gradients, f"No gradients for {name}"
        assert all(
            torch.isfinite(g).all().item()
            for g in gradients
        ), f"Non-finite gradients for {name}"

        nonzero = any(
            torch.count_nonzero(g).item() > 0
            for g in gradients
        )

        assert nonzero, f"All gradients are zero for {name}"

        print(f"{name}: gradients OK")

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=10.0
    )

    optimizer.step()

    print("\nOPTIMIZER STEP PASSED")


if __name__ == "__main__":
    main()
