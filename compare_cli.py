#!/usr/bin/env python
"""CLI entry point for comparing detected objects between two images.

Usage:
    python compare_cli.py --image1 photo_a.jpg --image2 photo_b.jpg [options]
"""

from compare import compare_images, print_comparison
import argparse
import json
import os
import sys


def main():
    execution_path = os.getcwd()

    parser = argparse.ArgumentParser(
        description="Compare detected objects between two images"
    )
    parser.add_argument(
        "--image1",
        type=str,
        required=True,
        help="Path to the first image",
    )
    parser.add_argument(
        "--image2",
        type=str,
        required=True,
        help="Path to the second image",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"),
        help="Path to the model file (default: resnet50_coco_best_v2.0.1.h5)",
    )
    parser.add_argument(
        "--confidence",
        type=int,
        default=30,
        help="Minimum confidence percentage for detections (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save comparison result as JSON",
    )
    args = parser.parse_args()

    for img in (args.image1, args.image2):
        if not os.path.isfile(img):
            print(f"Error: Image '{img}' does not exist.")
            sys.exit(1)

    result = compare_images(args.image1, args.image2, args.model, args.confidence)
    print_comparison(result)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nComparison saved to '{args.output}'.")


if __name__ == "__main__":
    main()
