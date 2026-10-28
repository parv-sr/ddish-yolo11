import torch
from torch import Tensor

from src.model import DDishYOLO


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = DDishYOLO(
        variant="n",
        p2_detection=True,
    ).to(device)

    model.train()

    images: Tensor = torch.rand(
        (2, 3, 256, 256),
        device=device,
    )

    predictions = model(images)

    nuclei = predictions["nuclei"]
    signals = predictions["signals"]

    assert isinstance(nuclei, dict)
    assert isinstance(signals, dict)

    print(
        "Nucleus features:",
        [tuple(x.shape) for x in nuclei["feats"]],
    )

    print(
        "Signal features:",
        [tuple(x.shape) for x in signals["feats"]],
    )

    print(
        "Nucleus scores:",
        tuple(nuclei["scores"].shape),
    )

    print(
        "Signal scores:",
        tuple(signals["scores"].shape),
    )

    assert len(nuclei["feats"]) == 3
    assert len(signals["feats"]) == 4

    assert nuclei["scores"].shape == (2, 1, 1344)
    assert signals["scores"].shape == (2, 3, 5440)

    assert torch.equal(
        model.signal_head.stride.cpu(),
        torch.tensor([4.0, 8.0, 16.0, 32.0]),
    )

    probe: Tensor = (
        nuclei["scores"].square().mean()
        + nuclei["proto"].square().mean()
        + signals["scores"].square().mean()
    )

    probe.backward()

    modules: dict[str, torch.nn.Module] = {
        "Shared network": model.shared,
        "P2 fusion": model.p2_fusion,
        "Nucleus head": model.nucleus_head,
        "Signal head": model.signal_head,
    }

    for name, module in modules.items():
        valid_gradients = [
            parameter.grad
            for parameter in module.parameters()
            if (
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all().item()
            )
        ]

        assert valid_gradients, f"No gradients in {name}"

        print(f"{name}: gradients OK")

    total_params: int = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Parameters: {total_params:,}")
    print("\nP2 MODEL SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
