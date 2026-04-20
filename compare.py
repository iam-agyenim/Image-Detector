from detection import detect
import os


def compare_images(image1, image2, model_path, min_confidence=30):
    """Compare detected objects between two images.

    Returns a dict with common objects, objects unique to each image,
    and a similarity score based on overlapping object categories.
    """
    dummy1 = f".tmp_cmp1_{os.getpid()}.jpg"
    dummy2 = f".tmp_cmp2_{os.getpid()}.jpg"

    try:
        detections1 = detect(image1, dummy1, model_path, min_confidence)
        detections2 = detect(image2, dummy2, model_path, min_confidence)
    finally:
        for path in (dummy1, dummy2):
            if os.path.exists(path):
                os.remove(path)

    objects1 = {d["name"] for d in detections1}
    objects2 = {d["name"] for d in detections2}

    common = objects1 & objects2
    only_in_first = objects1 - objects2
    only_in_second = objects2 - objects1
    all_objects = objects1 | objects2

    similarity = len(common) / len(all_objects) if all_objects else 1.0

    counts1 = {}
    for d in detections1:
        counts1[d["name"]] = counts1.get(d["name"], 0) + 1

    counts2 = {}
    for d in detections2:
        counts2[d["name"]] = counts2.get(d["name"], 0) + 1

    return {
        "image1": os.path.basename(image1),
        "image2": os.path.basename(image2),
        "image1_objects": counts1,
        "image2_objects": counts2,
        "common_objects": sorted(common),
        "only_in_image1": sorted(only_in_first),
        "only_in_image2": sorted(only_in_second),
        "similarity_score": round(similarity, 4),
    }


def print_comparison(result):
    print("=" * 50)
    print("  Image Comparison Results")
    print("=" * 50)
    print(f"  Image 1 : {result['image1']}")
    print(f"  Image 2 : {result['image2']}")
    print("-" * 50)
    print(f"  Objects in image 1:")
    for obj, count in result["image1_objects"].items():
        print(f"    {obj:20s}  count={count}")
    print(f"  Objects in image 2:")
    for obj, count in result["image2_objects"].items():
        print(f"    {obj:20s}  count={count}")
    print("-" * 50)
    print(f"  Common objects     : {', '.join(result['common_objects']) or 'None'}")
    print(f"  Only in image 1    : {', '.join(result['only_in_image1']) or 'None'}")
    print(f"  Only in image 2    : {', '.join(result['only_in_image2']) or 'None'}")
    print(f"  Similarity score   : {result['similarity_score']}")
    print("=" * 50)
