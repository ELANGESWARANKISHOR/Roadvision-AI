from ultralytics import YOLO


# Load YOLO model
model = YOLO("yolo11n.pt")


def detect_vehicles(image_path, confidence=0.25):
    """
    Detect vehicles in an image.

    Parameters:
        image_path: Path to the input image
        confidence: Minimum confidence threshold

    Returns:
        Dictionary containing detected vehicle counts
    """

    results = model.predict(
        source=image_path,
        conf=confidence,
        verbose=False
    )

    vehicle_counts = {
        "car": 0,
        "truck": 0,
        "bus": 0,
        "motorcycle": 0
    }

    for result in results:

        for class_id in result.boxes.cls:

            class_name = model.names[int(class_id)]

            if class_name in vehicle_counts:
                vehicle_counts[class_name] += 1

    return vehicle_counts


if __name__ == "__main__":

    image_path = "data/test/road.jpg"

    counts = detect_vehicles(image_path)

    print("Vehicle Detection Results")
    print("-------------------------")

    for vehicle, count in counts.items():
        print(f"{vehicle}: {count}")