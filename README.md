# Object Detection using ResNet50 and ImageAI

This project implements object detection using the pre-trained **ResNet50** model, leveraging the **ImageAI** library. The goal is to detect various objects in images with high accuracy using deep learning techniques. The repository includes all necessary files, such as the model, Python scripts, and required dependencies.

## Features

- Object detection using a pre-trained **ResNet50** model.
- Easy-to-use implementation with **ImageAI** library.
- Supports multiple object detection in images.
- Adjustable confidence level for detected objects.

## Requirements

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
absl-py==2.1.0
astunparse==1.6.3
certifi==2024.8.30
charset-normalizer==3.3.2
flatbuffers==24.3.25
gast==0.6.0
google-pasta==0.2.0
grpcio==1.66.1
h5py==3.11.0
idna==3.10
imageai==3.0.3
keras==3.5.0
libclang==18.1.1
Markdown==3.7
markdown-it-py==3.0.0
MarkupSafe==2.1.5
mdurl==0.1.2
ml-dtypes==0.3.2
namex==0.0.8
numpy==1.26.4
opencv-python==4.10.0.84
opt-einsum==3.3.0
optree==0.12.1
packaging==24.1
protobuf==4.25.4
Pygments==2.18.0
requests==2.32.3
rich==13.8.1
setuptools==75.1.0
six==1.16.0
tensorboard==2.16.2
tensorboard-data-server==0.7.2
tensorflow==2.16.2
termcolor==2.4.0
typing_extensions==4.12.2
urllib3==2.2.3
Werkzeug==3.0.4
wheel==0.44.0
wrapt==1.16.0

 ```

## FILES

- `detection.py`: Main Python script that handles object detection.
- `resnet50_coco_best_v2.0.1.h5`: Pre-trained ResNet50 model weights.
- `requirements.txt`: List of dependencies needed for the project.

## Download File

- `resnet50_coco_best_v2.0.1.h5` file: [Download here](https://drive.google.com/file/d/1olD0BRJl1JLtdU-z5QWk2Ki2c-HHri-R/view?usp=sharing)

## How to Use

1. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Place the image you want to process in the same directory as the script or specify the correct path.

3. Run the object detection script:

    ```bash
    python detection.py --input <path_to_image> --output <path_to_output_image>
    ```

    Replace `<path_to_image>` with the path to your input image and `<path_to_output_image>` with where you want to save the result.

4. Adjust confidence levels or other parameters in the `detection.py` script as needed.

## Pre-trained Model

The ResNet50 model is pre-trained on the COCO dataset. The model file included in this repository (`resnet50_coco_best_v2.0.1.h5`) enables accurate detection of a variety of object categories.

## License

This project is licensed under the MIT License. Feel free to use it and modify it as per your needs.

## Acknowledgments

- **ImageAI** for the detection framework.
- **COCO Dataset** for providing the dataset used to train the model.
