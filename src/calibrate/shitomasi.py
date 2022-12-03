import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("Screenshot_20221121_095116.png")
img = cv.imread("vlcsnap-2022-11-21-10h47m39s028.png")
hsv = cv.cvtColor(img,cv.COLOR_RGB2LAB)
h,s,v = cv.split(hsv)
corners = cv.goodFeaturesToTrack(v, 50, 0.04, 10)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv.circle(v,(x,y),3,255,-1)
plt.imshow(v),plt.show()