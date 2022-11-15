import cv2 as cv
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from icecream import ic

matplotlib.use('TkAgg')

max_value = 255
max_value_H = 255
low_H = 0
low_S = 0
low_V = 0
high_H = max_value
high_S = max_value
high_V = max_value
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)

cv.namedWindow(window_detection_name)
cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)


# cap = cv.VideoCapture("2022-11-11 16-00-12.mp4")

r = cv.imread("rough.png")
s = cv.imread("smooth.png")
rl = cv.imread("rough light.png")
sl = cv.imread("smooth light.png")

# thres = cv.inRange(r, (low_H, low_S, low_V), (high_H, high_S, high_V))
orig = s
luv = cv.cvtColor(orig, cv.COLOR_BGR2LUV)
l,u,v = cv.split(luv)
r,g,b = cv.split(orig)

# cv.imshow("l", l)
# cv.imshow("2", g)
# cv.imshow("3", v)


thres = cv.inRange(luv,(0, 0, 0),(210, 255, 110))
thresint = cv.cvtColor(thres, cv.COLOR_GRAY2BGR)
cnts, _ = cv.findContours(thres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnt=cnts[0]
for c in cnts:
    if(cv.moments(c)['m00'] > cv.moments(cnt)['m00']):
        cnt = c #find largest contour
x,y,w,h = cv.boundingRect(cnt) #get bounding box
croppedThres = thres[y: y+h, x: x+h]
croppedR = r[y: y+h, x: x+h]
# thres = cv.dilate(thres, None)
thres = cv.GaussianBlur(thres, (5,5), 0, 0)
thres = thres.astype('float64')
thresMax = np.max(abs(thres))
thres *= 1/thresMax

# thresBGR = cv.cvtColor(thres, cv.COLOR_GRAY2BGR)
det = cv.equalizeHist(croppedR)
# masked = cv.bitwise_and(thres, r) #extract R channel for the red reflection


laplacian = cv.Laplacian(det, cv.CV_64F)
sobel = cv.Sobel(det, cv.CV_64F, 1, 1)

masked = croppedThres * sobel


f = np.fft.fft2(masked)
cutoff = (np.divide(masked.shape, 5).astype('int'))#discard lower freqs
ic(cutoff)
ic(masked.shape)
fmask = np.zeros(f.shape)
for y in range(cutoff[0], fmask.shape[0]- cutoff[0]):
    for x in range(cutoff[1], fmask.shape[1] - cutoff[1]):
        fmask[y, x] = 1

ff = np.multiply(f, fmask)
# ff = f

# ic(f)
# ic(fmask)
# ic(ff)

roughness = np.sum(np.abs(ff))

ic(roughness)


fshift = np.fft.fftshift(ff)
# for y in range(np.shape(fshift)[0]):
#     for x in range(np.shape(fshift))
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.subplot(121),plt.imshow(masked, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

cv.imshow("mask", thres)
cv.imshow("orig", cv.bitwise_and(thresint, orig))
# cv.imshow("masked", masked)


while True:   
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()