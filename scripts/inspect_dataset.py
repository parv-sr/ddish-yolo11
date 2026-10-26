from pathlib import Path
from collections import Counter
from PIL import Image
import numpy as np
import json

RAW = Path("data/raw")

for slide in sorted(RAW.iterdir()):
    if not slide.is_dir():
        continue

    print(f"\n{'=' * 60}")
    print(f"DATASET: {slide.name}")
    print(f"{'=' * 60}")

    patch_dir = slide / "patches"
    mask_dir = slide / "Masks"
    annotation_dir = slide / "initial_annotations"

    patches = sorted(patch_dir.rglob("*.png"))
    masks = sorted(mask_dir.rglob("*.png"))
    annotations = sorted(annotation_dir.rglob("*.json"))

    print(f"Image patches: {len(patches)}")
    print(f"Nucleus masks: {len(masks)}")
    print(f"JSON files:    {len(annotations)}")

    print("\nExample patch filenames:")
    for p in patches[:3]:
        print(" ", p.name)

    print("\nExample mask filenames:")
    for p in masks[:3]:
        print(" ", p.name)

    # Check correspondence by filename
    patch_names = {p.stem for p in patches}
    mask_names = {p.stem for p in masks}

    print(f"\nMatching filenames: {len(patch_names & mask_names)}")

    # Inspect sample masks
    print("\nMask pixel inspection:")

    for path in masks[:5]:
        with Image.open(path) as image:
            arr = np.array(image)

            print(f"\n  {path.name}")
            print(f"  Shape: {arr.shape}")
            print(f"  Mode:  {image.mode}")
            print(f"  Dtype: {arr.dtype}")
            print(f"  Min:   {arr.min()}")
            print(f"  Max:   {arr.max()}")
            print(f"  Nonzero pixels: {np.count_nonzero(arr)}")
            print(f"  Unique values: {np.unique(arr)[:20]}")


    for path in annotations[:3]:
        with open(path) as f:
            data = json.load(f)

        print(f"\nAnnotation file: {path.name}")

        if "images" in data and "annotations" in data:
            print("  Images:", len(data["images"]))
            print("  Annotations:", len(data["annotations"]))

            categories = {
                c["id"]: c["name"]
                for c in data.get("categories", [])
            }

            counts = Counter(
                a["category_id"]
                for a in data["annotations"]
            )

            for category_id, name in categories.items():
                print(f"  {name}: {counts[category_id]}")
        else:
            print("  Non-COCO JSON structure")
