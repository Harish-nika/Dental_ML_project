from pathlib import Path

import cv2
import numpy as np
import requests
import supervision as sv


def load_image(image_source: str) -> np.ndarray:
    if image_source.startswith(("http://", "https://")):
        response = requests.get(image_source, timeout=60)
        response.raise_for_status()
        image_array = np.frombuffer(response.content, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(image_source)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_source}")
    return image


def annotate_and_save(
    image: np.ndarray,
    results: dict,
    output_path: Path,
    display: bool = False,
) -> Path:
    detections = sv.Detections.from_inference(results)

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated_image)

    if display:
        sv.plot_image(annotated_image)

    return output_path
