import cv2
from ultralytics import YOLO


model = YOLO("yolo11n.pt")


VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle"
}


def analyze_traffic_flow(input_path, output_path, confidence=0.25):

    video = cv2.VideoCapture(input_path)

    if not video.isOpened():
        raise ValueError("Could not open video")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    output = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    # Horizontal counting line
    line_y = height // 2

    counted_ids = set()

    frame_count = 0

    while True:

        success, frame = video.read()

        if not success:
            break

        results = model.track(
            frame,
            conf=confidence,
            persist=True,
            verbose=False
        )

        result = results[0]

        if result.boxes.id is not None:

            tracking_ids = result.boxes.id.tolist()
            class_ids = result.boxes.cls.tolist()
            boxes = result.boxes.xyxy.tolist()

            for tracking_id, class_id, box in zip(
                tracking_ids,
                class_ids,
                boxes
            ):

                class_name = model.names[int(class_id)]

                if class_name not in VEHICLE_CLASSES:
                    continue

                vehicle_id = int(tracking_id)

                x1, y1, x2, y2 = box

                center_y = (y1 + y2) / 2

                if center_y > line_y:

                    if vehicle_id not in counted_ids:

                        counted_ids.add(vehicle_id)

        annotated_frame = result.plot()

        cv2.line(
            annotated_frame,
            (0, line_y),
            (width, line_y),
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Vehicles crossed: {len(counted_ids)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        output.write(annotated_frame)

        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

    video.release()
    output.release()

    print("\nTraffic Flow Analysis")
    print("---------------------")
    print(f"Vehicles crossed line: {len(counted_ids)}")
    print(f"Frames processed: {frame_count}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":

    input_video = "data/test/traffic.mp4"
    output_video = "runs/traffic_flow.mp4"

    analyze_traffic_flow(
        input_video,
        output_video
    )