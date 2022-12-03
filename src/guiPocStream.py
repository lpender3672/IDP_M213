
import cv2 as cv
from icecream import ic
import numpy as np


def main():
    cap = cv.VideoCapture("2022-11-11 16-00-12.mp4")
    # ic(cap)
    while True:
        # ic(cap)
        ret, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow("frame", np.array(frame, dtype = np.uint8 ) )
        if cv.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    main()