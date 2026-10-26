"""COCO-backed paired nucleus-segmentation and gene-signal detection data."""

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

# Map the COCO category *names*, not numerical IDs (which may vary by export).
SIGNAL_CLASS_IDS = {
    "fusion(Her2+Chr17)": 0,
    "pink(Chr17)": 1,
    "black(Her2)": 2,
}


def load_records(raw_root="data/raw"):
    """Find all patches and their COCO labels across the three slide folders."""
    records = []
    raw_root = Path(raw_root)
    for slide in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        annotation_file = slide / "initial_annotations" / "instances_default.json"
        if not annotation_file.is_file():
            raise FileNotFoundError(annotation_file)
        coco = json.loads(annotation_file.read_text())
        names = {c["id"]: c["name"] for c in coco["categories"]}
        by_image = defaultdict(list)
        for ann in coco["annotations"]:
            if ann["category_id"] not in names:
                raise ValueError(f"Unknown category ID: {ann['category_id']}")
            by_image[ann["image_id"]].append(ann)

        for item in coco["images"]:
            filename = Path(item["file_name"]).name
            path = slide / "patches" / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            if "_y" not in filename:
                raise ValueError(f"Cannot determine source group: {filename}")
            records.append({
                "path": path,
                "group": filename.split("_y", 1)[0],
                "slide": slide.name,
                "height": int(item["height"]),
                "width": int(item["width"]),
                "annotations": by_image[item["id"]],
                "names": names,
            })
    if not records:
        raise ValueError(f"No images found under {raw_root}")
    return records


def split_by_source(records, seed=42):
    """Hold out one complete SL field from each slide folder."""
    rng = random.Random(seed)
    groups_by_slide = defaultdict(set)
    for record in records:
        groups_by_slide[record["slide"]].add(record["group"])

    held_out = set()
    for slide, groups in sorted(groups_by_slide.items()):
        if len(groups) < 2:
            raise ValueError(f"Need >=2 SL groups in {slide} to split by source")
        held_out.add(rng.choice(sorted(groups)))

    train = [r for r in records if r["group"] not in held_out]
    val = [r for r in records if r["group"] in held_out]
    return train, val, sorted(held_out)


def normalized_xywh(box, width, height):
    """COCO [left, top, width, height] -> YOLO [cx, cy, w, h] in [0, 1]."""
    x, y, w, h = map(float, box)
    if (w <= 0 or h <= 0 or x < 0 or y < 0
            or x + w > width + 0.01 or y + h > height + 0.01):
        raise ValueError(f"Invalid COCO box {box} for {width}x{height}")
    return [(x + w / 2) / width, (y + h / 2) / height,
            w / width, h / height]


class DDishDataset(Dataset):
    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        width, height = record["width"], record["height"]
        with Image.open(record["path"]) as image:
            image = image.convert("RGB")
            if image.size != (width, height):
                raise ValueError(f"Image dimensions disagree: {record['path']}")
            image_array = np.array(image, copy=True)
        pixels = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0

        nuclei_masks, nuclei_boxes = [], []
        signal_boxes, signal_classes = [], []
        for ann in record["annotations"]:
            name = record["names"][ann["category_id"]]
            if name == "Nucleus":
                canvas = Image.new("L", (width, height), 0)
                draw = ImageDraw.Draw(canvas)
                for polygon in ann.get("segmentation", []):
                    if len(polygon) >= 6 and len(polygon) % 2 == 0:
                        points = list(zip(polygon[::2], polygon[1::2]))
                        draw.polygon(points, fill=1)
                mask = np.array(canvas, dtype=np.uint8)
                if not mask.any():
                    raise ValueError(f"Empty nucleus polygon in {record['path']}, ann {ann['id']}")
                nuclei_masks.append(mask)
                nuclei_boxes.append(normalized_xywh(ann["bbox"], width, height))
            elif name in SIGNAL_CLASS_IDS:
                signal_boxes.append(normalized_xywh(ann["bbox"], width, height))
                signal_classes.append(SIGNAL_CLASS_IDS[name])
            # Omit exclude and the three sparse HER2 cluster classes.

        mask_tensor = (torch.from_numpy(np.stack(nuclei_masks)).bool()
                       if nuclei_masks else torch.zeros((0, height, width), dtype=torch.bool))

        def boxes_tensor(boxes):
            return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        return {
            "image": pixels,
            "nucleus_masks": mask_tensor,
            "nucleus_boxes": boxes_tensor(nuclei_boxes),
            "signal_boxes": boxes_tensor(signal_boxes),
            "signal_classes": torch.tensor(signal_classes, dtype=torch.long),
            "source_group": record["group"],
            "path": str(record["path"]),
        }


def collate_ddish(batch):
    """Stack images; keep variable-length per-image supervision separately."""
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "targets": [{k: v for k, v in item.items() if k != "image"} for item in batch],
    }
