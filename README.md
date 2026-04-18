# Object Detection using ResNet50 and ImageAI

This project implements object detection using the pre-trained **ResNet50** model, leveraging the **ImageAI** library. The goal is to detect various objects in images with high accuracy using deep learning techniques. The repository includes all necessary files, such as the model, Python scripts, and required dependencies.

## Features

- Object detection using a pre-trained **ResNet50** model.
- Easy-to-use implementation with **ImageAI** library.
- Supports multiple object detection in images.
- Adjustable confidence level for detected objects.
- JSON report generation for single images or entire directories.
- Programmatic image comparison to find common and unique objects between two images.
- In-depth single-image analysis with confidence breakdown and distribution.

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

- `detection.py`: Main Python script that handles single-image object detection.
- `batch_detection.py`: Script for batch object detection on a directory of images.
- `report.py`: Script to generate a JSON detection report for one or more images.
- `compare.py`: Module for comparing detected objects between two images.
- `stats.py`: Script to compute aggregate statistics from a JSON detection report.
- `analyzer.py`: Script for in-depth analysis of a single image's detections.
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

4. Adjust the minimum confidence threshold (default: 30%):

    ```bash
    python detection.py --input im.jpeg --output result.jpg --confidence 50
    ```

5. Use a custom model path:

    ```bash
    python detection.py --input im.jpeg --output result.jpg --model /path/to/model.h5
    ```

## Batch Detection

Process an entire directory of images at once using `batch_detection.py`:

```bash
python batch_detection.py --input-dir ./images --output-dir ./output
```

All supported image formats (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`) in the input directory will be processed. Output images are saved with a `_detected` suffix.

### Options

| Flag            | Description                                | Default                            |
|-----------------|--------------------------------------------|------------------------------------|
| `--input-dir`   | Directory containing input images (required) | —                                |
| `--output-dir`  | Directory to save output images            | `./output`                         |
| `--model`       | Path to the model file                     | `resnet50_coco_best_v2.0.1.h5`    |
| `--confidence`  | Minimum confidence percentage              | `30`                               |

## JSON Report Generation

Generate a structured JSON report of detected objects without keeping annotated images:

```bash
# Single image
python report.py --input im.jpeg

# Directory of images
python report.py --input ./images --output results.json
```

The report is saved as a JSON file where each key is an image filename and the value is a list of detected objects with their confidence scores.

### Options

| Flag            | Description                                   | Default                            |
|-----------------|-----------------------------------------------|------------------------------------|
| `--input`       | Path to an image or directory (required)       | —                                 |
| `--output`      | Path to save the JSON report                  | `report.json`                      |
| `--model`       | Path to the model file                        | `resnet50_coco_best_v2.0.1.h5`    |
| `--confidence`  | Minimum confidence percentage                 | `30`                               |

### Example Output

```json
{
  "im.jpeg": [
    {"name": "person", "confidence": 92.45},
    {"name": "dog", "confidence": 78.12}
  ]
}
```

## Report Statistics

Compute aggregate statistics from a JSON report generated by `report.py`:

```bash
python stats.py --input report.json
```

Optionally save the statistics as a JSON file:

```bash
python stats.py --input report.json --output stats.json
```

### Options

| Flag       | Description                                      | Default         |
|------------|--------------------------------------------------|-----------------|
| `--input`  | Path to the JSON report from `report.py`         | `report.json`   |
| `--output` | Optional path to save statistics as JSON         | —               |

### Example Output

```
==================================================
  Detection Report Statistics
==================================================
  Total images analysed  : 3
  Total detections       : 12
  Unique object types    : 4
  Avg detections / image : 4.0
  Most common object     : person
  Least common object    : dog
--------------------------------------------------
  Object counts:
    person                count=   6  avg_confidence=87.32%
    car                   count=   3  avg_confidence=74.55%
    bicycle               count=   2  avg_confidence=68.90%
    dog                   count=   1  avg_confidence=78.12%
==================================================
```

## Image Comparison

Compare detected objects between two images programmatically using `compare.py`:

```python
from compare import compare_images

result = compare_images(
    "image1.jpeg",
    "image2.jpeg",
    "resnet50_coco_best_v2.0.1.h5",
    min_confidence=30,
)
print(result)
```

### Return Value

The `compare_images` function returns a dictionary with:

| Key                | Description                                              |
|--------------------|----------------------------------------------------------|
| `image1`           | Basename of the first image                              |
| `image2`           | Basename of the second image                             |
| `image1_objects`   | Object names and counts detected in the first image      |
| `image2_objects`   | Object names and counts detected in the second image     |
| `common_objects`   | Object categories found in both images                   |
| `only_in_image1`   | Object categories found only in the first image          |
| `only_in_image2`   | Object categories found only in the second image         |
| `similarity_score` | Ratio of shared categories to total categories (0.0–1.0) |

### Example Output

```python
{
    "image1": "park.jpeg",
    "image2": "street.jpeg",
    "image1_objects": {"person": 3, "dog": 1},
    "image2_objects": {"person": 2, "car": 4},
    "common_objects": ["person"],
    "only_in_image1": ["dog"],
    "only_in_image2": ["car"],
    "similarity_score": 0.3333
}
```

## Image Analyzer

Perform an in-depth analysis of detected objects in a single image, including per-object confidence statistics and an overall confidence distribution:

```bash
python analyzer.py --input im.jpeg
```

Optionally save the analysis as a JSON file:

```bash
python analyzer.py --input im.jpeg --output analysis.json
```

### Options

| Flag            | Description                                | Default                            |
|-----------------|--------------------------------------------|------------------------------------|
| `--input`       | Path to the input image (required)         | —                                  |
| `--model`       | Path to the model file                     | `resnet50_coco_best_v2.0.1.h5`    |
| `--confidence`  | Minimum confidence percentage              | `30`                               |
| `--output`      | Optional path to save analysis as JSON     | —                                  |

### Example Output

```
=======================================================
  Image Analysis
=======================================================
  Image                : im.jpeg
  Total objects found  : 5
  Unique object types  : 3
  Overall confidence   : min=34.12%  max=95.67%  mean=72.45%
-------------------------------------------------------
  Object breakdown:
    person                count=  3  min= 45.2%  max= 95.7%  mean= 74.3%
    car                   count=  1  min= 82.1%  max= 82.1%  mean= 82.1%
    dog                   count=  1  min= 34.1%  max= 34.1%  mean= 34.1%
-------------------------------------------------------
  Confidence distribution:
    low (30-50%)           1  #
    medium (50-75%)        1  #
    high (75-100%)         3  ###
-------------------------------------------------------
  High-confidence detections (>=75%):
    - person (95.67%)
    - car (82.1%)
    - person (76.55%)
  Low-confidence detections (<50%):
    - dog (34.12%)
=======================================================
```

### Programmatic Usage

```python
from analyzer import analyze_image

analysis = analyze_image("im.jpeg", "resnet50_coco_best_v2.0.1.h5", min_confidence=30)
print(analysis)
```

The `analyze_image` function returns a dictionary with:

| Key                         | Description                                                  |
|-----------------------------|--------------------------------------------------------------|
| `image`                     | Basename of the analysed image                               |
| `total_objects`             | Total number of detected objects                             |
| `unique_objects`            | Number of distinct object categories                         |
| `object_counts`             | Object names and their counts                                |
| `overall_confidence`        | Min, max, and mean confidence across all detections          |
| `confidence_stats`          | Per-object min, max, mean confidence and count               |
| `confidence_distribution`   | Number of detections in low / medium / high confidence bands |
| `high_confidence_objects`   | Detections with confidence ≥ 75%                             |
| `low_confidence_objects`    | Detections with confidence < 50%                             |

## Pre-trained Model

The ResNet50 model is pre-trained on the COCO dataset. The model file included in this repository (`resnet50_coco_best_v2.0.1.h5`) enables accurate detection of a variety of object categories.

## License

This project is licensed under the MIT License. Feel free to use it and modify it as per your needs.

## Acknowledgments

- **ImageAI** for the detection framework.
- **COCO Dataset** for providing the dataset used to train the model.
