import time

import cv2
from ultralytics import YOLO
from tracker import Tracker


model = YOLO('yolov8n.pt') # Load a pretrained YOLO model (you can choose n, s, m, l, or x versions)

#VIDEO_PATH = 'data/footage.mp4'     # swap this file path for footage 
VIDEO_PATH = 0                     # swap this for webcam use

def main():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Video was not opened at {VIDEO_PATH}")
        return

    tracker = Tracker()
    last_print = 0.0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Video ended")
            break

        
        results = model.track(frame, persist=True, classes=[0], verbose=False)
        tracker.update(results, frame)   # frame needed so the tracker can crop people for re-ID
        
        if time.time() - last_print >= 1.0:
            tracker.print_all()
            last_print = time.time()
        #print(results[0].boxes)
        annotated_frame = results[0].plot()
        cv2.imshow('Cloud Collar', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()