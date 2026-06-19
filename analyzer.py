from detection import detect
from collections import Counter
import argparse
import json
import os
import sys


def analyze_image(input_image, model_path, min_confidence=30):
    """Perform detailed analysis of detected objects in a single image.

    Returns a dictionary with object counts, confidence breakdown,
    detection density, and per-object statistics.
    """
    dummy_output = f".tmp_analyze_{os.getpid()}.jpg"

    try:
        detections = detect(input_image, dummy_output, model_path, min_confidence)
    finally:
        if os.path.exists(dummy_output):
            os.remove(dummy_output)

    if not detections:
        return {
            "image": os.path.basename(input_image),
            "total_objects": 0,
            "unique_objects": 0,
            "object_counts": {},
            "confidence_stats": {},
            "confidence_distribution": {},
            "high_confidence_objects": [],
            "low_confidence_objects": [],
        }

    names = [d["name"] for d in detections]
    object_counts = dict(Counter(names).most_common())

    confidence_by_object = {}
    for d in detections:
        confidence_by_object.setdefault(d["name"], []).append(
            d["percentage_probability"]
        )

    confidence_stats = {}
    for name, confs in confidence_by_object.items():
        confidence_stats[name] = {
            "min": round(min(confs), 2),
            "max": round(max(confs), 2),
            "mean": round(sum(confs) / len(confs), 2),
            "count": len(confs),
        }

    all_confs = [d["percentage_probability"] for d in detections]
    overall_mean = round(sum(all_confs) / len(all_confs), 2)
    overall_min = round(min(all_confs), 2)
    overall_max = round(max(all_confs), 2)

    buckets = {"low (30-50%)": 0, "medium (50-75%)": 0, "high (75-100%)": 0}
    for c in all_confs:
        if c < 50:
            buckets["low (30-50%)"] += 1
        elif c < 75:
            buckets["medium (50-75%)"] += 1
        else:
            buckets["high (75-100%)"] += 1

    high_conf = sorted(
        [d for d in detections if d["percentage_probability"] >= 75],
        key=lambda x: x["percentage_probability"],
        reverse=True,
    )
    low_conf = sorted(
        [d for d in detections if d["percentage_probability"] < 50],
        key=lambda x: x["percentage_probability"],
    )

    return {
        "image": os.path.basename(input_image),
        "total_objects": len(detections),
        "unique_objects": len(object_counts),
        "object_counts": object_counts,
        "overall_confidence": {
            "min": overall_min,
            "max": overall_max,
            "mean": overall_mean,
        },
        "confidence_stats": confidence_stats,
        "confidence_distribution": buckets,
        "high_confidence_objects": [
            {"name": d["name"], "confidence": round(d["percentage_probability"], 2)}
            for d in high_conf
        ],
        "low_confidence_objects": [
            {"name": d["name"], "confidence": round(d["percentage_probability"], 2)}
            for d in low_conf
        ],
    }


def print_analysis(analysis):
    print("=" * 55)
    print("  Image Analysis")
    print("=" * 55)
    print(f"  Image                : {analysis['image']}")
    print(f"  Total objects found  : {analysis['total_objects']}")
    print(f"  Unique object types  : {analysis['unique_objects']}")

    if analysis["total_objects"] == 0:
        print("  No objects detected.")
        print("=" * 55)
        return

    overall = analysis["overall_confidence"]
    print(f"  Overall confidence   : min={overall['min']}%  max={overall['max']}%  mean={overall['mean']}%")
    print("-" * 55)
    print("  Object breakdown:")
    for name, stats in analysis["confidence_stats"].items():
        print(
            f"    {name:20s}  count={stats['count']:3d}"
            f"  min={stats['min']:5.1f}%  max={stats['max']:5.1f}%  mean={stats['mean']:5.1f}%"
        )
    print("-" * 55)
    print("  Confidence distribution:")
    for bucket, count in analysis["confidence_distribution"].items():
        bar = "#" * count
        print(f"    {bucket:18s}  {count:3d}  {bar}")
    print("-" * 55)

    if analysis["high_confidence_objects"]:
        print("  High-confidence detections (>=75%):")
        for obj in analysis["high_confidence_objects"]:
            print(f"    - {obj['name']} ({obj['confidence']}%)")
    else:
        print("  No high-confidence detections (>=75%).")

    if analysis["low_confidence_objects"]:
        print("  Low-confidence detections (<50%):")
        for obj in analysis["low_confidence_objects"]:
            print(f"    - {obj['name']} ({obj['confidence']}%)")
    else:
        print("  No low-confidence detections (<50%).")

    print("=" * 55)


def main():
    execution_path = os.getcwd()

    parser = argparse.ArgumentParser(
        description="Analyze detected objects in an image with detailed statistics"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input image",
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
        help="Optional path to save the analysis as JSON",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    analysis = analyze_image(args.input, args.model, args.confidence)
    print_analysis(analysis)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis saved to '{args.output}'.")


if __name__ == "__main__":
    main()
