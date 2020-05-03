from threading import Thread
from queue import Queue
from typing import List
import numpy as np
import cv2
import time
from .utils import get_bird_keywords
from .model import load_model
from .detect import detect_from_frame, record_when_keyword


class Detection(Thread):
    def __init__(self, model, q: Queue, out_q: Queue, keywords: List = None):
        self.model = model
        self.q = q
        self.out_q = out_q
        self.is_detecting = False
        self.keywords = keywords or []
        super().__init__()

    def run(self) -> None:
        self.is_detecting = True
        while self.is_detecting:
            frame = self.q.get(timeout=5)
            rankings = detect_from_frame(self.model, frame)
            if self.out_q.empty():
                self.out_q.put(record_when_keyword(rankings, keywords=self.keywords))
            print("-------------")
            for name, conf in rankings[:10]:
                print(f"{conf * 100:10.2f}\t{name}")
            print("-------------")
            time.sleep(3)


def webcam_detect(record: bool = False, if_bird: bool = True):
    model = load_model()
    queue = Queue(maxsize=1)
    out_q = Queue(maxsize=1)
    detection = Detection(model, queue, out_q, keywords=get_bird_keywords() if if_bird else [""])
    detection.start()

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    if record and rval:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        h, w = frame.shape[:-1]
        out = cv2.VideoWriter('recording.avi', fourcc, 15, (w, h), True)

    keyword_present = False

    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        if queue.empty():
            queue.put(frame)
        if record:
            if out_q.full():
                keyword_present = out_q.get()
            if keyword_present:
                write_to_video(out, frame)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            detection.is_detecting = False
            break
    cv2.destroyWindow("preview")
    out.release()


def write_to_video(out, frame):
    h, w = frame.shape[:-1]
    output = np.zeros((h, w, 3), dtype="uint8")
    output[0:h, 0:w] = frame
    out.write(output)
