#test
#https://github.com/Thyix/astar-pathfinding


import math
import cv2 as cv
import numpy as np
from icecream import ic
from timeit import default_timer as timer

# cap = cv.VideoCapture("http://localhost:8081/stream/video.mjpeg")
cap = cv.VideoCapture("2022-11-10 09-09-04.mp4")
cap = cv.VideoCapture("2022-11-10 09-07-51.mp4")


# https://github.com/kaustubh-sadekar/VirtualCam/blob/master/GUI.py

def nothing(x):
    pass


window_detection_name = 'Object Detection'

# https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html

max_value = 255
max_value_H = 180
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

on_low_V_thresh_trackbar(228)



width  = 1012
height = 760

distCoeff = np.zeros((4,1),np.float64)

# TODO: add your coefficients here!
k1 = -2.5e-5; # negative to remove barrel distortion
k2 = 0.0;
p1 = 25e-5;
p2 = 0.0;

distCoeff[0,0] = k1;
distCoeff[1,0] = k2;
distCoeff[2,0] = p1;
distCoeff[3,0] = p2;

# assume unit matrix for camera
cam = np.eye(3,dtype=np.float32)

cam[0,2] = width/2.0  # define center x
cam[1,2] = height/2.0 # define center y
cam[0,0] = 6.        # define focal length x
cam[1,1] = 6.        # define focal length y

pathCornerStack = []

# cap = cv.undistort(cap,cam,distCoeff)



while True:

    start = timer()
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = frame[0:height, 0:width]
    
    # X = -cv.getTrackbarPos("X",WINDOW_NAME) + 500
    # Y = -cv.getTrackbarPos("Y",WINDOW_NAME) + 500
    # Z = -cv.getTrackbarPos("Z",WINDOW_NAME)

    # k1 = (cv.getTrackbarPos("K1",WINDOW_NAME)-5)/100000
    # k2 = (cv.getTrackbarPos("K2",WINDOW_NAME)-5)/100000
    # p1 = cv.getTrackbarPos("P1",WINDOW_NAME)/100000
    # p2 = cv.getTrackbarPos("P2",WINDOW_NAME)/100000

    distCoeff[0,0] = k1;
    distCoeff[1,0] = k2;
    distCoeff[2,0] = p1;
    distCoeff[3,0] = p2;

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    dist = cv.undistort(frame,cam,distCoeff)


    # https://github.com/ShaheerSajid/OpenCV-Maze-Solving/blob/main/code/Source.cpp
    hsv = cv.cvtColor(dist, cv.COLOR_BGR2HSV) #src BGR to HSV	
    frame_threshold = cv.inRange(hsv, (0, 0, 228), (255, 255, 255))
    frame_threshold = cv.GaussianBlur(frame_threshold, (9, 9), 1);
    kernel = np.ones((19, 19), np.uint8)
    dilation = cv.dilate(frame_threshold, kernel, iterations=1) 
    erosion = cv.erode(dilation, kernel, iterations=1)
    erosionG = cv.Canny(erosion, 40, 200, None, 3)
    # erosionG2 = cv.erode(erosion, np.ones((3,3), np.uint8), iterations = 1)
    distO = np.copy(dist)
    # distO = np.zeros((height,width,3), np.uint8)
    

    rotTheta = 0
    rotM = cv.getRotationMatrix2D((width/2, height/2), rotTheta * 180 / np.pi, 1.0)

    lines = cv.HoughLines(erosionG, 1, np.pi / 250, 175, None, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(distO, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
            try:
                rotTheta += (pt1[1]-pt2[1])/(pt1[0]-pt2[0])
            except ZeroDivisionError:
                ic("Rotation Lock Lost")

        rotTheta = rotTheta / len(lines)
        rotTheta = math.atan(rotTheta)

        rotM = cv.getRotationMatrix2D((width/2, height/2), rotTheta * 180 / np.pi, 0.8)

    distOL = cv.warpAffine(src=distO, M=rotM, dsize=(width, height))
    distOR = cv.warpAffine(src=dist, M=rotM, dsize=(width, height))

    
    dim = (int(width/2), int(height/2))
    distOR = cv.resize(distOR, dim, interpolation = cv.INTER_AREA)


    corrected = cv.cvtColor(distOR, cv.COLOR_BGR2HSV)
    tresh_path = cv.inRange(corrected, (0, 0, 190), (255, 16, 255))
    tresh_G = cv.inRange(corrected, (66, 50, 70), (85, 255, 255))
    correctedINV = cv.cvtColor(cv.bitwise_not(distOR), cv.COLOR_BGR2HSV)
    # rconv = cv.cvtColor(correctedRGB, cv.COLOR_BGR2LAB)
    # rconv = cv.cvtColor(correctedRGB, cv.COLOR_BGR2YCrCb)
    rconv = correctedINV.copy() #HSV 
    (hr,sr,vr) = cv.split(rconv)
    # ic(rconv)
    hr = np.add(hr, 30)
    rconv = cv.merge((hr, sr, vr))
    # tresh_R = cv.inRange(rconv, (110, 108, 101), (183, 186, 139))  #use LAB
    # tresh_R = cv.inRange(rconv, (0, 144, 105), (255, 212, 162)) #use ycbcr
    tresh_R = cv.inRange(rconv, (95, 49, 50), (126, 181, 255)) #use HSV
    # tresh_R = cv.inRange(rconv, (low_H, low_S, low_V), (high_H, high_S, high_V))
    tresh_O = cv.inRange(corrected, (27, 91, 182), (52, 255, 255))

    
    tresh_R = cv.GaussianBlur(tresh_R, (5,5), 1);
    rkernel = np.ones((3,3), np.uint8)
    tresh_R = cv.erode(tresh_R, None, iterations = 1)
    tresh_R = cv.dilate(tresh_R, rkernel, iterations=1) 
    # tresh_R = cv.erode(tresh_R, rkernel, iterations=1)

    tresh_path = cv.GaussianBlur(tresh_path, (15, 15), 1);
    rkernel = np.ones((3,3), np.uint8)    
    tresh_path = cv.erode(tresh_path, rkernel, iterations=1)
    tresh_path = cv.dilate(tresh_path, rkernel, iterations=1) 

    tresh_G = cv.GaussianBlur(tresh_G, (15,15), 1);
    rkernel = np.ones((5,5), np.uint8)
    tresh_G = cv.erode(tresh_G, rkernel, iterations=1)
    tresh_G = cv.dilate(tresh_G, rkernel, iterations=1) 
    

    tresh_O = cv.GaussianBlur(tresh_O, (37, 37), 1);
    rkernel = np.ones((11, 11), np.uint8)
    tresh_O = cv.dilate(tresh_O, rkernel, iterations=1) 
    tresh_O = cv.erode(tresh_O, rkernel, iterations=1)

    #get a diagnostic image
    map = tresh_path.copy()
    map = cv.cvtColor(map, cv.COLOR_GRAY2BGR)

    # find green square
    cnts,_ = cv.findContours(tresh_G.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    gc = cv.cvtColor(tresh_G, cv.COLOR_GRAY2BGR)
    cv.drawContours(gc, cnts, -1, (0,255,0), 2)
    for cnt in cnts:
        M = cv.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv.circle(gc, (cx,cy), radius=3, color=(0, 255, 0), thickness=-1)
        map = cv.circle(map, (cx,cy), radius=3, color=(0, 255, 0), thickness=-1)

    #find red square
    cnts,hie = cv.findContours(tresh_R.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    rc = cv.cvtColor(tresh_R, cv.COLOR_GRAY2BGR)
    cv.drawContours(rc, cnts, -1, (0,255,0), 2)
    for cnt in cnts:
        M = cv.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv.circle(rc, (cx,cy), radius=3, color=(0, 0, 255), thickness=-1)
        map = cv.circle(map, (cx,cy), radius=3, color=(0, 0, 255), thickness=-1)

    
    #find tunnel
    cnts,hie = cv.findContours(tresh_O.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    oc = cv.cvtColor(tresh_O, cv.COLOR_GRAY2BGR)
    cv.drawContours(oc, cnts, -1, (0,255,0), 2)


    #find endpoints on path
    # po = cv.Canny(tresh_path, 100, 200, None, 3)
    # tresh_pathB = cv.GaussianBlur(tresh_path, (3), 1)
    tresh_pathF = np.float32(tresh_path)
    pathCorner = cv.cornerHarris(tresh_pathF, 5, 3, 0.04)
    pathCorner = cv.erode(pathCorner, None, iterations = 2)
    pathCorner = cv.dilate(pathCorner,None)

    #do temporal filtering
    ntmax = 15    
    if(len(pathCornerStack)) == ntmax:
        pathCornerStack.pop(-1) #circular buffer
    pathCornerStack.append(pathCorner)
    pathCornerStackS = np.sum(pathCornerStack, axis = 0)
    pathCornerStackF = np.where(pathCornerStackS>255*12, 255, 0)
    # ic(pathCornerStackF)
    pathCornerStackF = pathCornerStackF.astype(np.uint8)


    #find points that are within the obstacle so we can interpolate
    tresh_OD = cv.dilate(tresh_O, None, iterations = 5)
    cntsO,_ = cv.findContours(tresh_OD.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cntsP,_ = cv.findContours(pathCornerStackF.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    pc = cv.cvtColor(pathCornerStackF, cv.COLOR_GRAY2BGR)
    oc = cv.cvtColor(tresh_OD, cv.COLOR_GRAY2BGR)
    cv.drawContours(pc, cntsP, -1, (0,255,0), 1)
    obsEndPoint = []
    for conO in cntsO:
        for conP in cntsP:
            mu = cv.moments(conP)
            (px, py) = (int(mu['m10'] / (mu['m00'] + 1e-5)), int(mu['m01'] / (mu['m00'] + 1e-5)))
            d = cv.pointPolygonTest(conO, (px, py), False)
            # ic(dist)
            if d == 1.0:
                obsEndPoint.append((px, py))
    # ic(obsEndPoint)
    for EP in obsEndPoint:
        # ic(EP)
        oc = cv.circle(oc, EP, radius=1, color=(0, 0, 255), thickness=-1)
        map = cv.circle(map, EP, radius=3, color=(0, 255, 255), thickness=-1)
    

    rcv = cv.split(rconv)    

    # Display the resulting frame
    cv.imshow('frame', tresh_R)
    # cv.imshow("red0", rcv[0])
    # cv.imshow("red1", rcv[1])
    # cv.imshow("red2", rcv[2])
    cv.imshow("map", map)
    end = timer()
    # ic(end-start)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
