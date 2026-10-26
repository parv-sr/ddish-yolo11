from pathlib import Path
from collections import defaultdict, Counter
import json
import numpy as np
from PIL import Image, ImageDraw

RAW = Path("data/raw")
OUT = Path("runs/inspection")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "pink(Chr17)": (255, 70, 190),
    "black(Her2)": (40, 230, 90),
    "fusion(Her2+Chr17)": (255, 210, 40),
}

for slide in sorted(RAW.iterdir()):
    if not slide.is_dir():
        continue

    annotation_path = (
        slide / "initial_annotations" / "instances_default.json"
    )

    with open(annotation_path) as f:
        coco = json.load(f)

    categories = {
        c["id"]: c["name"] for c in coco["categories"]
    }

    annotations = defaultdict(list)
    for ann in coco["annotations"]:
        annotations[ann["image_id"]].append(ann)

    mismatches = []
    valid_images = []

    sources = Counter(
        Path(info["file_name"]).stem.split("_y")[0]
        for info in coco["images"]
    )

    for info in coco["images"]:
        filename = Path(info["file_name"]).name
        patch_path = slide / "patches" / filename
        mask_path = slide / "Masks" / filename

        if not patch_path.exists() or not mask_path.exists():
            print("Missing image or mask:", filename)
            continue

        with Image.open(mask_path) as f:
            mask = np.array(f)

        ids = np.unique(mask)
        instance_count = np.count_nonzero(ids)

        coco_count = sum(
            categories[a["category_id"]] == "Nucleus"
            for a in annotations[info["id"]]
        )

        if instance_count != coco_count:
            mismatches.append(
                (filename, instance_count, coco_count)
            )

        valid_images.append(info)

    print(f"\n{slide.name}")
    print("Source identifiers:", dict(sources))
    print("Matched images:", len(valid_images))
    print("Nucleus-count mismatches:", len(mismatches))
    print("First five mismatches:", mismatches[:5])

    if not valid_images:
        continue

    def counts(info):
        anns = annotations[info["id"]]
        nucleus = sum(
            categories[a["category_id"]] == "Nucleus"
            for a in anns
        )
        signals = len(anns) - nucleus
        return nucleus, signals

    sample = next(
        (
            info for info in valid_images
            if min(counts(info)) >= 8
        ),
        valid_images[0],
    )

    filename = Path(sample["file_name"]).name

    image = Image.open(
        slide / "patches" / filename
    ).convert("RGB")

    with Image.open(slide / "Masks" / filename) as f:
        mask = np.array(f)

    overlay = np.array(image, dtype=np.float32)
    rng = np.random.default_rng(42)

    for instance_id in np.unique(mask):
        if instance_id == 0:
            continue

        color = rng.integers(60, 256, size=3)
        pixels = mask == instance_id

        overlay[pixels] = (
            0.65 * overlay[pixels] + 0.35 * color
        )

    result = Image.fromarray(overlay.astype(np.uint8))
    draw = ImageDraw.Draw(result)

    for ann in annotations[sample["id"]]:
        name = categories[ann["category_id"]]

        if name in ("Nucleus", "exclude"):
            continue

        x, y, w, h = ann["bbox"]

        draw.rectangle(
            (x, y, x + w, y + h),
            outline=COLORS.get(name, (255, 255, 255)),
            width=2,
        )

    comparison = Image.new("RGB", (image.width * 2, image.height))
    comparison.paste(image, (0, 0))
    comparison.paste(result, (image.width, 0))

    comparison.resize(
        (image.width * 4, image.height * 2),
        Image.Resampling.NEAREST,
    ).save(OUT / f"{slide.name}_overlay.png")

    print("Saved overlay for:", filename)

print("\nInspection complete.")
