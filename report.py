from detection import detect
import argparse
import json
import os
import sys


def generate_report(input_image, output_image, model_path, report_path, min_confidence=30):
    if not os.path.isfile(input_image):
        print(f"Error: Input image '{input_image}' does not exist.")
        sys.exit(1)

    detections = detect(input_image, output_image, model_path, min_confidence)

    report = {
        "input_image": os.path.abspath(input_image),
        "output_image": os.path.abspath(output_image),
        "model": os.path.abspath(model_path),
        "min_confidence": min_confidence,
        "total_objects": len(detections),
        "detections": [
            {
                "name": obj["name"],
                "confidence": round(obj["percentage_probability"], 2),
            }
            for obj in detections
        ],
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {report_path}")
    return report


def main():
    execution_path = os.getcwd()

    parser = argparse.ArgumentParser(
        description="Run object detection and export results as a JSON report"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(execution_path, "imnew.jpg"),
        help="Path to save the output image (default: imnew.jpg)",
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
        "--report",
        type=str,
        default=os.path.join(execution_path, "report.json"),
        help="Path to save the JSON report (default: report.json)",
    )
    args = parser.parse_args()

    generate_report(args.input, args.output, args.model, args.report, args.confidence)


if __name__ == "__main__":
    main()
