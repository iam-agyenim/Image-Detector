from imageai.Detection import ObjectDetection
import os

# Get the current working directory
execution_path = os.getcwd()

# Create an instance of the ObjectDetection class
detector = ObjectDetection()

# Set model type as RetinaNet
detector.setModelTypeAsRetinaNet()

# Set the path to the model
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))

# Load the model
detector.loadModel()

# Perform object detection
detections = detector.detectObjectsFromImage(
    input_image=os.path.join(execution_path, "im.jpeg"),
    output_image_path=os.path.join(execution_path, "imnew.jpg")
)

# Print out detected objects and their probability
for eachObject in detections:
    print(eachObject["name"],
           ":",
             eachObject["percentage_probability"]
    )
