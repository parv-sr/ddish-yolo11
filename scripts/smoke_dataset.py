"""Run from the project root: uv run python scripts/smoke_dataset.py"""
from torch.utils.data import DataLoader
from src.dataset import DDishDataset, collate_ddish, load_records, split_by_source

records = load_records("data/raw")
train, val, held_out = split_by_source(records, seed=42)
assert not ({r["group"] for r in train} & {r["group"] for r in val})
print(f"Total: {len(records)}, train: {len(train)}, val: {len(val)}")
print("Validation SL groups:", held_out)

train_set = DDishDataset(train)
loader = DataLoader(train_set, batch_size=4, shuffle=False,
                    num_workers=0, collate_fn=collate_ddish)
batch = next(iter(loader))
print("Images:", tuple(batch["images"].shape), batch["images"].dtype)
for target in batch["targets"]:
    print(target["source_group"],
          "nuclei:", tuple(target["nucleus_masks"].shape),
          "signals:", tuple(target["signal_boxes"].shape),
          "classes:", sorted(set(target["signal_classes"].tolist())))
print("SMOKE TEST PASSED")
