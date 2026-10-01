import cv2
from ultralytics import YOLO


model = YOLO("yolo11n.pt")


def process_video(input_path, output_path, confidence=0.25):

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

    frame_count = 0

    while True:

        success, frame = video.read()

        if not success:
            break

        results = model.predict(
            frame,
            conf=confidence,
            verbose=False
        )

        annotated_frame = results[0].plot()

        output.write(annotated_frame)

        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

    video.release()
    output.release()

    print(f"\nVideo processing completed.")
    print(f"Total frames processed: {frame_count}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":

    input_video = "data/test/traffic.mp4"
    output_video = "runs/traffic_output.mp4"

    process_video(
        input_video,
        output_video
    )