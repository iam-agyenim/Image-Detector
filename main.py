import argparse
import os
import sys


def cmd_detect(args):
    from detection import detect

    detect(args.input, args.output, args.model, args.confidence)


def cmd_batch(args):
    from batch_detection import batch_detect

    batch_detect(args.input_dir, args.output_dir, args.model, args.confidence)


def cmd_report(args):
    from report import generate_report

    generate_report(args.input, args.model, args.output, args.confidence)


def cmd_stats(args):
    from stats import load_report, compute_stats, print_stats
    import json

    report = load_report(args.input)
    stats = compute_stats(report)
    print_stats(stats)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to '{args.output}'.")


def cmd_compare(args):
    from compare import compare_images
    import json

    result = compare_images(args.image1, args.image2, args.model, args.confidence)
    print(json.dumps(result, indent=2))


def main():
    execution_path = os.getcwd()
    default_model = os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5")

    parser = argparse.ArgumentParser(
        description="Image Detector — object detection toolkit powered by ImageAI and ResNet50"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # detect
    p_detect = subparsers.add_parser("detect", help="Detect objects in a single image")
    p_detect.add_argument("--input", type=str, default=os.path.join(execution_path, "im.jpeg"),
                          help="Path to the input image (default: im.jpeg)")
    p_detect.add_argument("--output", type=str, default=os.path.join(execution_path, "imnew.jpg"),
                          help="Path to save the output image (default: imnew.jpg)")
    p_detect.add_argument("--model", type=str, default=default_model,
                          help="Path to the model file")
    p_detect.add_argument("--confidence", type=int, default=30,
                          help="Minimum confidence percentage (default: 30)")
    p_detect.set_defaults(func=cmd_detect)

    # batch
    p_batch = subparsers.add_parser("batch", help="Batch detect objects in a directory of images")
    p_batch.add_argument("--input-dir", type=str, required=True,
                         help="Directory containing input images")
    p_batch.add_argument("--output-dir", type=str, default=os.path.join(execution_path, "output"),
                         help="Directory to save output images (default: ./output)")
    p_batch.add_argument("--model", type=str, default=default_model,
                         help="Path to the model file")
    p_batch.add_argument("--confidence", type=int, default=30,
                         help="Minimum confidence percentage (default: 30)")
    p_batch.set_defaults(func=cmd_batch)

    # report
    p_report = subparsers.add_parser("report", help="Generate a JSON detection report")
    p_report.add_argument("--input", type=str, required=True,
                          help="Path to an image file or directory of images")
    p_report.add_argument("--output", type=str, default=os.path.join(execution_path, "report.json"),
                          help="Path to save the JSON report (default: report.json)")
    p_report.add_argument("--model", type=str, default=default_model,
                          help="Path to the model file")
    p_report.add_argument("--confidence", type=int, default=30,
                          help="Minimum confidence percentage (default: 30)")
    p_report.set_defaults(func=cmd_report)

    # stats
    p_stats = subparsers.add_parser("stats", help="Show statistics from a JSON detection report")
    p_stats.add_argument("--input", type=str, default="report.json",
                         help="Path to the JSON report (default: report.json)")
    p_stats.add_argument("--output", type=str, default=None,
                         help="Optional path to save statistics as JSON")
    p_stats.set_defaults(func=cmd_stats)

    # compare
    p_compare = subparsers.add_parser("compare", help="Compare detected objects between two images")
    p_compare.add_argument("image1", type=str, help="Path to the first image")
    p_compare.add_argument("image2", type=str, help="Path to the second image")
    p_compare.add_argument("--model", type=str, default=default_model,
                           help="Path to the model file")
    p_compare.add_argument("--confidence", type=int, default=30,
                           help="Minimum confidence percentage (default: 30)")
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
