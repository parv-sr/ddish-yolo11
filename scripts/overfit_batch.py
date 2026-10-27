import torch
from src.dataset import (
    DDishDataset,
    SIGNAL_CLASS_IDS,
    collate_ddish,
    load_records,
    split_by_source,
)
from src.model import DDishYOLO
from src.losses import DDishLoss


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load the training split only.
    records = load_records("data/raw")
    train_records, _, _ = split_by_source(records, seed=42)

    # Pick two images containing several nuclei and signals.
    # Prefer multiple signal classes so both tasks have
    # useful supervision.
    def suitable(record):
        names = record["names"]
        labels = [
            names[ann["category_id"]]
            for ann in record["annotations"]
        ]

        nuclei = labels.count("Nucleus")
        signals = [
            name for name in labels
            if name in SIGNAL_CLASS_IDS
        ]

        return (
            nuclei >= 3
            and len(signals) >= 3
            and len(set(signals)) >= 2
        )

    selected = [r for r in train_records if suitable(r)][:2]

    if len(selected) != 2:
        raise RuntimeError("Could not find two suitable images")

    dataset = DDishDataset(selected)

    # Load ONCE. We will reuse this exact batch every step.
    batch = collate_ddish([dataset[0], dataset[1]])

    images = batch["images"].to(device)
    targets = batch["targets"]

    print("Device:", device)
    print("Batch:", tuple(images.shape))

    for target in targets:
        print(
            target["path"],
            "nuclei:", len(target["nucleus_masks"]),
            "signals:", len(target["signal_boxes"]),
            "classes:", target["signal_classes"].tolist(),
        )

    model = DDishYOLO(variant="n").to(device)
    model.train()

    criterion = DDishLoss(model)

    # No weight decay: we WANT to memorize these images.
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=5e-4,
        weight_decay=0.0,
    )

    steps = 100
    batch_size = images.shape[0]
    history = []

    print("\nStarting fixed-batch overfitting test\n")
    print("Step | Total/img | Seg/img | Det/img")
    print("-" * 44)

    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)

        predictions = model(images)
        loss, details = criterion(predictions, targets)

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Non-finite loss at step {step}: {loss.item()}"
            )

        total = loss.item() / batch_size
        seg = details["segmentation"].item() / batch_size
        det = details["detection"].item() / batch_size

        history.append((total, seg, det))

        if step % 10 == 0:
            print(
                f"{step:4d} | "
                f"{total:9.4f} | "
                f"{seg:7.4f} | "
                f"{det:7.4f}"
            )

        # Step 100 measures the result AFTER 100 updates.
        if step == steps:
            break

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=10.0
        )

        optimizer.step()

    initial = history[0]
    final = history[-1]

    print("\nLoss change:")
    for name, before, after in zip(
        ("Total", "Segmentation", "Detection"),
        initial,
        final,
    ):
        change = 100 * (after - before) / max(before, 1e-12)
        print(f"{name:15s}: {before:.4f} -> {after:.4f} "
              f"({change:+.1f}%)")

    if final[0] < initial[0]:
        print("\nTotal loss decreased.")
    else:
        print("\nWARNING: Total loss did not decrease.")

    if final[1] >= initial[1] or final[2] >= initial[2]:
        print("WARNING: Check individual task losses.")


if __name__ == "__main__":
    main()
