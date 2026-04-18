from imageai.Detection import ObjectDetection
import argparse
import os


def detect(input_image, output_image, model_path, min_confidence=30):
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(model_path)
    detector.loadModel()

    detections = detector.detectObjectsFromImage(
        input_image=input_image,
        output_image_path=output_image,
        minimum_percentage_probability=min_confidence,
    )

    for eachObject in detections:
        print(eachObject["name"], ":", eachObject["percentage_probability"])

    return detections


def main():
    execution_path = os.getcwd()

    parser = argparse.ArgumentParser(description="Object detection using ResNet50 and ImageAI")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(execution_path, "im.jpeg"),
        help="Path to the input image (default: im.jpeg)",
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
    args = parser.parse_args()

    detect(args.input, args.output, args.model, args.confidence)


if __name__ == "__main__":
    main()
