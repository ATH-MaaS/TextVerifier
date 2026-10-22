from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from datasets import Image, load_dataset
from PIL import Image as PILImage

EXPECTED_IMAGE_IDS = [f"{index:03d}" for index in range(308)]
EXPECTED_IMAGE_COLUMNS = {"image_id", "image", "image_path", "width", "height"}
EXPECTED_ANNOTATION_COLUMNS = {
    "sample_id",
    "image_id",
    "image_path",
    "item_index",
    "char",
    "box",
    "rect",
    "render_error",
}
EXPECTED_STATS = {
    "images": 308,
    "character_crops": 7358,
    "render_errors": 642,
    "render_correct": 6716,
}


def load_parquet(dataset_dir: Path):
    image_files = sorted((dataset_dir / "data" / "images").glob("test-*.parquet"))
    annotation_files = sorted((dataset_dir / "data" / "annotations").glob("test-*.parquet"))
    if not image_files or not annotation_files:
        raise FileNotFoundError("dataset directory must contain both test Parquet configurations")
    images = load_dataset("parquet", data_files={"test": [str(path) for path in image_files]}, split="test")
    annotations = load_dataset("parquet", data_files={"test": [str(path) for path in annotation_files]}, split="test")
    if set(images.column_names) != EXPECTED_IMAGE_COLUMNS:
        raise ValueError(f"unexpected images schema: {images.column_names}")
    if set(annotations.column_names) != EXPECTED_ANNOTATION_COLUMNS:
        raise ValueError(f"unexpected annotations schema: {annotations.column_names}")
    return images.cast_column("image", Image(decode=False)), annotations


def is_han(char: str) -> bool:
    return len(char) == 1 and (
        0x4E00 <= ord(char) <= 0x9FFF
        or 0x3400 <= ord(char) <= 0x4DBF
        or 0x20000 <= ord(char) <= 0x2EBEF
        or 0xF900 <= ord(char) <= 0xFAFF
    )


def validate_images(images) -> dict[str, bytes]:
    if len(images) != 308:
        raise ValueError(f"expected 308 images, found {len(images)}")
    records = {}
    for row in images:
        image_id = row["image_id"]
        image = row["image"]
        payload = image.get("bytes") if isinstance(image, dict) else None
        if image_id in records or not isinstance(payload, bytes):
            raise ValueError(f"invalid image record: {image_id}")
        if row["image_path"] != f"images/{image_id}.png" or row["width"] != 1024 or row["height"] != 1024:
            raise ValueError(f"unexpected image metadata: {image_id}")
        with PILImage.open(io.BytesIO(payload)) as decoded:
            if decoded.format != "PNG" or decoded.mode != "RGB" or decoded.size != (1024, 1024):
                raise ValueError(f"unexpected image format: {image_id}")
            decoded.verify()
        records[image_id] = payload
    if sorted(records) != EXPECTED_IMAGE_IDS:
        raise ValueError("image IDs must be 000 through 307")
    return records


def validate_annotations(annotations, images: dict[str, bytes]) -> list[dict]:
    if len(annotations) != 7358:
        raise ValueError(f"expected 7358 annotations, found {len(annotations)}")
    rows = []
    seen = set()
    for row in annotations:
        box = [[float(value) for value in point] for point in row["box"]]
        rect = [float(value) for value in row["rect"]]
        rebuilt = {
            "sample_id": str(row["sample_id"]),
            "image_id": str(row["image_id"]),
            "image": str(row["image_path"]),
            "item_index": int(row["item_index"]),
            "char": str(row["char"]),
            "box": box,
            "rect": rect,
            "render_error": bool(row["render_error"]),
        }
        if rebuilt["sample_id"] in seen or rebuilt["item_index"] < 0 or not is_han(rebuilt["char"]):
            raise ValueError(f"invalid annotation: {rebuilt['sample_id']}")
        if rebuilt["image_id"] not in images or rebuilt["image"] != f"images/{rebuilt['image_id']}.png":
            raise ValueError(f"invalid annotation image reference: {rebuilt['sample_id']}")
        if len(box) != 4 or any(len(point) != 2 for point in box) or len(rect) != 4:
            raise ValueError(f"invalid geometry: {rebuilt['sample_id']}")
        coordinates = [value for point in box for value in point] + rect
        x0, y0, x1, y1 = rect
        if not all(math.isfinite(value) for value in coordinates) or x1 <= x0 or y1 <= y0:
            raise ValueError(f"invalid geometry: {rebuilt['sample_id']}")
        seen.add(rebuilt["sample_id"])
        rows.append(rebuilt)
    if sum(row["render_error"] for row in rows) != 642:
        raise ValueError("expected 642 rendering errors")
    return sorted(rows, key=lambda row: row["sample_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize TextRenderBench-ZH Parquet files for TextVerifier evaluation.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="local root of the downloaded Hugging Face Dataset repository")
    parser.add_argument("--output", type=Path, default=Path("benchmark"), help="new benchmark directory to create")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    images, annotations = load_parquet(dataset_dir)
    image_bytes = validate_images(images)
    rows = validate_annotations(annotations, image_bytes)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        image_dir = staging / "images"
        image_dir.mkdir()
        for image_id in EXPECTED_IMAGE_IDS:
            (image_dir / f"{image_id}.png").write_bytes(image_bytes[image_id])
        with (staging / "annotations.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        (staging / "stats.json").write_text(json.dumps(EXPECTED_STATS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({**EXPECTED_STATS, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
