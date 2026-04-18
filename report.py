from detection import detect
import argparse
import json
import os
import sys

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


def generate_report(input_path, model_path, output_report, min_confidence=30):
    results = {}

    if os.path.isfile(input_path):
        images = [input_path]
    elif os.path.isdir(input_path):
        images = [
            os.path.join(input_path, f)
            for f in sorted(os.listdir(input_path))
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ]
        if not images:
            print(f"No supported images found in '{input_path}'.")
            sys.exit(0)
    else:
        print(f"Error: '{input_path}' is not a valid file or directory.")
        sys.exit(1)

    for image_path in images:
        name = os.path.basename(image_path)
        dummy_output = os.path.join(
            os.path.dirname(image_path),
            f".tmp_report_{os.getpid()}.jpg",
        )

        print(f"Analyzing: {name}")
        detections = detect(image_path, dummy_output, model_path, min_confidence)

        results[name] = [
            {"name": d["name"], "confidence": round(d["percentage_probability"], 2)}
            for d in detections
        ]

        if os.path.exists(dummy_output):
            os.remove(dummy_output)

    with open(output_report, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nReport saved to '{output_report}'.")
    return results


def main():
    execution_path = os.getcwd()

    parser = argparse.ArgumentParser(
        description="Generate a JSON detection report for one or more images"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to an image file or a directory of images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(execution_path, "report.json"),
        help="Path to save the JSON report (default: report.json)",
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
    args = parser.parse_args()

    generate_report(args.input, args.model, args.output, args.confidence)


if __name__ == "__main__":
    main()
