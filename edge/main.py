import os
import time

os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

import cv2
from ultralytics import YOLO
from tracker import Tracker
from uploader import upload

UPLOAD_INTERVAL = 30  # seconds between periodic uploads

model = YOLO('yolov8s.pt')        # person tracking
phone_model = YOLO('yolov8s.pt')  # separate instance — avoids interfering with ByteTrack state

VIDEO_PATH = 'data/testingfootage_fixed.MOV'     # swap this file path for footage
#VIDEO_PATH = 0                                  # swap this for webcam use

SKIP_SECONDS = 10     # ignore this many seconds at the start (camera setup, etc.)

COLORS = [
    (255, 80,  80),
    (80,  80,  255),
    (80,  255, 80),
    (255, 180, 0),
    (180, 0,   255),
    (0,   200, 255),
    (255, 140, 0),
    (100, 255, 220),
]


def _phone_inside_person(phone_box, person_boxes) -> bool:
    """Returns True if the phone box center falls inside any person box."""
    px = (phone_box[0] + phone_box[2]) / 2
    py = (phone_box[1] + phone_box[3]) / 2
    for pb in person_boxes:
        if pb[0] <= px <= pb[2] and pb[1] <= py <= pb[3]:
            return True
    return False


def main():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Video was not opened at {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    skip_frames = int(SKIP_SECONDS * fps)
    if skip_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
        print(f"Skipping first {SKIP_SECONDS}s ({skip_frames} frames)")

    tracker = Tracker()
    last_print = 0.0
    last_upload = 0.0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Video ended")
            upload(tracker)  # final upload on exit
            break

        results = model.track(frame, persist=True, classes=[0], conf=0.6, verbose=False)

        # separate inference pass for phones — no tracking needed, just boxes
        phone_results = phone_model.predict(frame, classes=[67], conf=0.25, verbose=False)
        all_phone_boxes = phone_results[0].boxes.xyxy.cpu().numpy().astype(int) if phone_results[0].boxes else []

        # only keep phones that overlap a person box — filters out phones sitting on counters
        person_boxes_np = results[0].boxes.xyxy.cpu().numpy().astype(int) if results[0].boxes.id is not None else []
        phone_boxes = [pb for pb in all_phone_boxes if _phone_inside_person(pb, person_boxes_np)]

        tracker.update(results, phone_boxes, frame)

        now = time.time()

        if now - last_print >= 1.0:
            tracker.print_all()
            last_print = now

        if now - last_upload >= UPLOAD_INTERVAL:
            upload(tracker)
            last_upload = now

        annotated_frame = frame.copy()
        boxes = results[0].boxes
        if boxes.id is not None:
            for box, track_id in zip(boxes.xyxy.cpu().numpy().astype(int), boxes.id.tolist()):
                person_id = tracker.track_to_person.get(int(track_id))
                color = COLORS[(person_id - 1) % len(COLORS)] if person_id else (180, 180, 180)
                label = f"person {person_id}" if person_id else "pending..."
                x1, y1, x2, y2 = box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 4)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for box in phone_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(annotated_frame, "phone", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        display = cv2.resize(annotated_frame, (1280, 720))
        cv2.imshow('Cloud Collar', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            upload(tracker)  # final upload on manual quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
