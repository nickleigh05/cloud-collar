import os
import time

os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

import cv2
from ultralytics import YOLO
from tracker import Tracker

UPLOAD_INTERVAL = 30  # seconds between uploads

model = YOLO('yolov8s.pt')        # person tracking
phone_model = YOLO('yolov8s.pt')  # phone tracking

VIDEO_PATH = 'data/testingfootage_fixed.MOV'        # swap this file path for footage
#VIDEO_PATH = 0                                     # swap this for webcam use

SKIP_SECONDS = 10     # change # to skip that many seconds into vid

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

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Video ended")
            break

        results = model.track(frame, persist=True, classes=[0], conf=0.6, verbose=False)

        # no tracking needed, just boxes
        phone_results = phone_model.predict(frame, classes=[67], conf=0.25, verbose=False)
        all_phone_boxes = phone_results[0].boxes.xyxy.cpu().numpy().astype(int) if phone_results[0].boxes else []

        # filters out phones sitting on counters
        person_boxes_np = results[0].boxes.xyxy.cpu().numpy().astype(int) if results[0].boxes.id is not None else []
        phone_boxes = [pb for pb in all_phone_boxes if _phone_inside_person(pb, person_boxes_np)]

        tracker.update(results, phone_boxes, frame)
        
        if time.time() - last_print >= 1.0:
            tracker.print_all()
            last_print = time.time()
            
        # draw people boxs in different colors
        COLORS = [
            (255, 80,  80),   # blue
            (80,  80,  255),  # red
            (80,  255, 80),   # green
            (255, 180, 0),    # cyan
            (180, 0,   255),  # magenta
            (0,   200, 255),  # yellow
            (255, 140, 0),    # teal
            (100, 255, 220),  # lime
        ]

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
                
        # draw phone boxes in white
        for box in phone_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(annotated_frame, "phone", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        display = cv2.resize(annotated_frame, (1280, 720))  #  display size
        cv2.imshow('Cloud Collar', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()