from detector import detect_vehicles


def calculate_traffic_density(vehicle_counts):
    """
    Calculate traffic density based on the total number
    of detected vehicles.
    """

    total_vehicles = sum(vehicle_counts.values())

    if total_vehicles <= 5:
        density = "LOW"

    elif total_vehicles <= 15:
        density = "MEDIUM"

    else:
        density = "HIGH"

    return {
        "total_vehicles": total_vehicles,
        "density": density
    }


if __name__ == "__main__":

    image_path = "data/test/road.jpg"

    detection_result = detect_vehicles(image_path)

    vehicle_counts = detection_result["counts"]

    traffic_result = calculate_traffic_density(vehicle_counts)

    print("Traffic Analysis")
    print("-----------------")
    print(f"Vehicle counts: {vehicle_counts}")
    print(f"Total vehicles: {traffic_result['total_vehicles']}")
    print(f"Traffic density: {traffic_result['density']}")