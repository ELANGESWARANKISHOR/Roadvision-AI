from ultralytics import YOLO


# Load YOLO model
model = YOLO("yolo11n.pt")


def detect_vehicles(image_path, confidence=0.25):
    """
    Detect vehicles and return detailed detection information.
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

    detections = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence_score = float(box.conf[0])

            if class_name in vehicle_counts:

                vehicle_counts[class_name] += 1

                coordinates = box.xyxy[0].tolist()

                detections.append({
                    "vehicle": class_name,
                    "confidence": round(confidence_score, 3),
                    "bounding_box": [
                        round(value, 2)
                        for value in coordinates
                    ]
                })

    return {
        "counts": vehicle_counts,
        "detections": detections
    }


if __name__ == "__main__":

    image_path = "data/test/road.jpg"

    result = detect_vehicles(image_path)

    print("Vehicle Detection Results")
    print("-------------------------")

    for vehicle, count in result["counts"].items():
        print(f"{vehicle}: {count}")

    print("\nIndividual Detections")
    print("---------------------")

    for detection in result["detections"]:
        print(detection)