import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import (
    DDishDataset,
    collate_ddish,
    load_records,
    split_by_source,
)
from src.losses import DDishLoss
from src.model import DDishYOLO


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/ddish11n"),
    )
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()


def make_loader(records, batch_size, shuffle, workers, pin_memory):
    return DataLoader(
        DDishDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_ddish,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


def run_epoch(model, criterion, loader, optimizer, device):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

        # Request raw predictions from the YOLO heads for loss
        # calculation, while keeping BatchNorm in evaluation mode.
        model.nucleus_head.training = True
        model.signal_head.training = True

    totals = np.zeros(3, dtype=np.float64)
    n_images = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            images = batch["images"].to(
                device, non_blocking=True
            )
            targets = batch["targets"]

            if training:
                optimizer.zero_grad(set_to_none=True)

            predictions = model(images)
            loss, parts = criterion(predictions, targets)

            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite loss encountered")

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 10.0
                )

                optimizer.step()

            # Ultralytics returns batch-scaled losses.
            totals += [
                loss.detach().item(),
                parts["segmentation"].item(),
                parts["detection"].item(),
            ]

            n_images += images.shape[0]

    return totals / n_images


def main():
    args = parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")

    args.run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    records = load_records("data/raw")
    train_records, val_records, held_out = split_by_source(
        records, seed=42
    )

    print("Device:", device)
    print(
        f"Train: {len(train_records)} / "
        f"Val: {len(val_records)}"
    )
    print("Held-out source groups:", held_out)

    train_loader = make_loader(
        train_records,
        args.batch_size,
        True,
        args.workers,
        device.type == "cuda",
    )

    val_loader = make_loader(
        val_records,
        args.batch_size,
        False,
        args.workers,
        device.type == "cuda",
    )

    model = DDishYOLO(variant="n").to(device)
    criterion = DDishLoss(model)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.1,
    )

    last_path = args.run_dir / "last.pt"
    best_path = args.run_dir / "best.pt"
    csv_path = args.run_dir / "history.csv"

    start_epoch = 1
    best_val = float("inf")

    if args.resume:
        checkpoint = torch.load(
            last_path,
            map_location=device,
            weights_only=True,
        )

        if checkpoint["held_out"] != held_out:
            raise RuntimeError("Training split mismatch")

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        start_epoch = checkpoint["epoch"] + 1
        best_val = checkpoint["best_val"]

        print(f"Resuming from epoch {start_epoch}")

    else:
        with csv_path.open("w", newline="") as file:
            csv.writer(file).writerow([
                "epoch",
                "lr",
                "train_total",
                "train_seg",
                "train_det",
                "val_total",
                "val_seg",
                "val_det",
                "seconds",
            ])

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()

        lr = optimizer.param_groups[0]["lr"]

        train_losses = run_epoch(
            model, criterion, train_loader, optimizer, device
        )

        val_losses = run_epoch(
            model, criterion, val_loader, None, device
        )

        scheduler.step()
        elapsed = time.perf_counter() - started

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train {train_losses[0]:.3f} "
            f"(seg {train_losses[1]:.3f}, "
            f"det {train_losses[2]:.3f}) | "
            f"val {val_losses[0]:.3f} "
            f"(seg {val_losses[1]:.3f}, "
            f"det {val_losses[2]:.3f}) | "
            f"{elapsed:.1f}s",
            flush=True,
        )

        with csv_path.open("a", newline="") as file:
            csv.writer(file).writerow([
                epoch,
                lr,
                *train_losses,
                *val_losses,
                elapsed,
            ])

        improved = val_losses[0] < best_val

        if improved:
            best_val = float(val_losses[0])

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val": best_val,
            "held_out": held_out,
            "variant": "n",
            "signal_classes": [
                "fusion(Her2+Chr17)",
                "pink(Chr17)",
                "black(Her2)",
            ],
        }

        torch.save(checkpoint, last_path)

        if improved:
            torch.save(checkpoint, best_path)

    print("\nTraining finished.")
    print("Best checkpoint:", best_path)


if __name__ == "__main__":
    main()
