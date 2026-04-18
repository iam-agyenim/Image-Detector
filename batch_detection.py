from detection import detect
import argparse
import os
import sys

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


def batch_detect(input_dir, output_dir, model_path, min_confidence=30):
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    images = [
        f for f in sorted(os.listdir(input_dir))
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    if not images:
        print(f"No supported images found in '{input_dir}'.")
        sys.exit(0)

    print(f"Found {len(images)} image(s) in '{input_dir}'.\n")

    for image_name in images:
        input_path = os.path.join(input_dir, image_name)
        name, _ = os.path.splitext(image_name)
        output_path = os.path.join(output_dir, f"{name}_detected.jpg")

        print(f"--- Processing: {image_name} ---")
        detect(input_path, output_path, model_path, min_confidence)
        print(f"    Saved to: {output_path}\n")

    print(f"Batch detection complete. Results saved in '{output_dir}'.")


def main():
    execution_path = os.getcwd()

    parser = argparse.ArgumentParser(
        description="Batch object detection on a directory of images"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(execution_path, "output"),
        help="Directory to save output images (default: ./output)",
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

    batch_detect(args.input_dir, args.output_dir, args.model, args.confidence)


if __name__ == "__main__":
    main()
