from ultralytics import YOLO
import cv2
import numpy as np

def get_frame_at_timestamp(video_path: str, time_sec: float):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_sec * fps))
    success, frame = cap.read()
    cap.release()
    if not success:
        raise RuntimeError(f"Could not read frame at {time_sec} seconds.")
    return frame

def detect_traffic_light_color(hsv_roi):
    # Only compare counts directly
    red_mask = cv2.inRange(hsv_roi, (0, 70, 50), (10, 255, 255)) | \
               cv2.inRange(hsv_roi, (160, 70, 50), (180, 255, 255))
    green_mask = cv2.inRange(hsv_roi, (40, 70, 50), (80, 255, 255))

    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)

    if red_pixels > green_pixels:
        return "RED"
    elif green_pixels > red_pixels:
        return "GREEN"
    else:
        return "UNKNOWN"

def analyze_frame_with_yolo(frame):
    model = YOLO("yolov8n.pt")  # or a better one like yolov8m.pt
    results = model.predict(source=frame, conf=0.3, verbose=False)[0]
    annotated = frame.copy()

    for box in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, class_id = box.astype(int)
        if class_id != 9:  # COCO class 9 = traffic light
            continue

        traffic_light_roi = frame[y1:y2, x1:x2]
        if traffic_light_roi.size == 0:
            continue

        # Focus only on the bottom part (light bulb)
        h_roi = traffic_light_roi.shape[0]
        bulb_roi = traffic_light_roi[int(h_roi * 0.6):, :]
        hsv = cv2.cvtColor(bulb_roi, cv2.COLOR_BGR2HSV)

        color = detect_traffic_light_color(hsv)

        # Draw result
        color_map = {"RED": (0, 0, 255), "GREEN": (0, 255, 0), "UNKNOWN": (128, 128, 128)}
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_map[color], 2)
        cv2.putText(annotated, color, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_map[color], 2)

        # Save cropped view
        cv2.imwrite("semaphore_zoomed_debug.jpg", traffic_light_roi)
        break  # Use first detected traffic light

    cv2.imwrite("annotated_frame.jpg", annotated)
    return color

# === MAIN USAGE ===
if __name__ == "__main__":
    video_path = "challenge_color_848x480.mp4"
    timestamp_sec = 3 * 60 + 56  # 3:56

    frame = get_frame_at_timestamp(video_path, timestamp_sec)
    cv2.imwrite("full_frame_3m56s.jpg", frame)

    result = analyze_frame_with_yolo(frame)
    print(f"🚦 Detected traffic light color at 3:56 → {result}")
