#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from PIL import Image
from PIL import ImageDraw

from scrfd import SCRFD, Threshold


def iter_images(in_dir: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for path in sorted(in_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def detect_on_image(detector: SCRFD, image_path: Path, threshold: Threshold):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        faces = detector.detect(img, threshold=threshold)

    draw = ImageDraw.Draw(img)
    for face in faces:
        x1 = face.bbox.upper_left.x
        y1 = face.bbox.upper_left.y
        x2 = face.bbox.lower_right.x
        y2 = face.bbox.lower_right.y
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        kps = face.keypoints
        points = [
            (kps.left_eye.x, kps.left_eye.y),
            (kps.right_eye.x, kps.right_eye.y),
            (kps.nose.x, kps.nose.y),
            (kps.left_mouth.x, kps.left_mouth.y),
            (kps.right_mouth.x, kps.right_mouth.y),
        ]
        r = 3
        for (x, y) in points:
            draw.ellipse((x - r, y - r, x + r, y + r), outline="red", fill="red")

    return img


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch face detection with SCRFD")
    parser.add_argument("--in_dir", type=Path, help="Directory containing input images")
    parser.add_argument("--out_dir", type=Path, help="Directory to write detection JSON files")
    parser.add_argument("--model_path", type=Path, default="./models/scrfd.onnx", help="Path to SCRFD ONNX model")
    parser.add_argument("--threshold", type=float, default=0.4, help="Detection probability threshold (0-1)")
    args = parser.parse_args()

    if not args.in_dir.is_dir():
        parser.error(f"in_dir does not exist or is not a directory: {args.in_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    detector = SCRFD.from_path(args.model_path)
    th = Threshold(probability=args.threshold)

    image_paths = list(iter_images(args.in_dir))
    if not image_paths:
        print(f"No images found in {args.in_dir}", file=sys.stderr)
        return 1

    for image_path in image_paths:
        try:
            annotated = detect_on_image(detector, image_path, threshold=th)
        except Exception as exc:  # pragma: no cover - convenience guard
            print(f"Failed on {image_path.name}: {exc}", file=sys.stderr)
            continue

        out_file = args.out_dir / image_path.name
        annotated.save(out_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
