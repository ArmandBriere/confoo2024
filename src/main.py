import argparse
from datetime import datetime
from random import randrange
from typing import List, Tuple

import cv2.dnn
import numpy as np
import yaml
from cv2.typing import MatLike

THRESHOLD = 0.5


def get_random_color() -> Tuple[int, int, int]:
    """Return a random color."""

    x: int = randrange(256)
    y: int = randrange(256)
    z: int = randrange(256)

    return (x, y, z)


def load_model(weights_file: str, cfg_file: str) -> cv2.dnn.DetectionModel:
    """Load yolo model with cv.dnn."""

    net = cv2.dnn.readNet(weights_file, cfg_file)

    model: cv2.dnn.DetectionModel = cv2.dnn.DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

    return model


def draw_boxes(
    original_image: np.ndarray,
    boxes: List[List[int]],
    detected_classes: List[int],
    classes: List[str],
    confidences: List[float],
):
    """Draw draws multiple boxes on the image."""
    for index, box in enumerate(boxes):
        draw_bounding_box(
            original_image, box, classes[detected_classes[index]], confidences[index]
        )


def draw_bounding_box(
    img: MatLike,
    box: List[float],
    class_name: str,
    confidence: float,
):
    """Draw bounding box and label for each detection."""
    x, y, w, h = box[0], box[1], box[2], box[3]

    # Get a random color
    random_color: Tuple[int, int, int] = get_random_color()

    # Prepare label
    label = f"{class_name} ({confidence:.2f})"

    # Draw rectangle and text
    cv2.rectangle(img, (x, y), (x + w, y + h), random_color, 2)
    cv2.putText(
        img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, random_color, 2
    )


def detect(net: cv2.dnn.DetectionModel, input_image: str, classes: List[str]):
    """Perform inference on the input image."""

    # Read the image
    original_image: np.ndarray = cv2.imread(input_image)

    now: datetime = datetime.now()

    # Perform inference
    detected_classes, scores, boxes = net.detect(original_image, THRESHOLD, 0.25)

    elapsed_time: float = (datetime.now() - now).total_seconds() * 1000
    print(f"{elapsed_time:.0f}")

    # # Apply NMS (Non-maximum suppression)
    # result_boxes = cv2.dnn.NMSBoxes(boxes, scores, THRESHOLD, 0.45, 0.25)

    draw_boxes(original_image, boxes, detected_classes, classes, scores)

    # Write the image to disk
    cv2.imwrite("python_inference.jpg", original_image)


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", default=str("person.jpg"), help="Path to input image.")
    args = parser.parse_args()

    # Load the model and classes
    net_model: cv2.dnn.DetectionModel = load_model(
        "yolov4-tiny.weights", "yolov4-tiny.cfg"
    )
    with open("classes.yml", "r") as file:
        config: dict = yaml.safe_load(file)
        classes: List[str] = config["names"]

    detect(net_model, args.img, classes)
