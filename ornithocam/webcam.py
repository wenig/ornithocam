from threading import Thread
from queue import Queue
from typing import List
import cv2
import time
from .model import load_model
from .detect import detect_from_frame


class Detection(Thread):
    def __init__(self, model, q: Queue):
        self.model = model
        self.q = q
        self.is_detecting = False
        super().__init__()

    def run(self) -> None:
        self.is_detecting = True
        while self.is_detecting:
            frame = self.q.get(timeout=5)
            rankings = detect_from_frame(self.model, frame)
            print("-------------")
            for name, conf in rankings[:10]:
                print(f"{conf * 100:10.2f}\t{name}")
            print("-------------")
            time.sleep(3)


def webcam_detect():
    model = load_model()
    queue = Queue(maxsize=1)
    detection = Detection(model, queue)
    detection.start()
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        if queue.empty():
            queue.put(frame)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            detection.is_detecting = False
            break
    cv2.destroyWindow("preview")


def record(frame, keywords: List):
    pass


if __name__ == "__main__":
    webcam_detect()