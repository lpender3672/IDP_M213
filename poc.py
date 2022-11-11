#test
#https://github.com/Thyix/astar-pathfinding


import math
import cv2 as cv
import numpy as np
from icecream import ic
from timeit import default_timer as timer

# cap = cv.VideoCapture("http://localhost:8081/stream/video.mjpeg")
# cap = cv.VideoCapture("2022-11-10 09-09-04.mp4")
# cap = cv.VideoCapture("2022-11-10 09-07-51.mp4")
# cap = cv.VideoCapture("2022-11-11 16-00-12.mp4")
cap = cv.VideoCapture("2022-11-11 16-33-38.mp4")


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

pathCornerStack = []


width  = 1012
height = 760

def acqFrame():
    ret, frame = cap.read()
    frame = frame[0:height, 0:width] 
    # ic("acq")     
    # if frame is read correctly ret is True
    if not ret:
        raise Exception("Can't receive frame (stream end?). Exiting ...")
    return frame

def acqCorrectedFrame(distCoeff, rotM, resizeFactor, blurSize):

    frame = acqFrame()
    dist = cv.undistort(frame,cam,distCoeff)
    distOR = cv.warpAffine(src=dist, M=rotM, dsize=(width, height))
    dim = (int(width/resizeFactor), int(height/resizeFactor))
    distOR = cv.GaussianBlur(distOR, (blurSize, blurSize), 1)
    distOR = cv.resize(distOR, dim, interpolation = cv.INTER_AREA)
    return distOR
       
def drawDiagnosticPoint(frame, point, color):
    return cv.circle(frame, point, radius=1, color=color, thickness=-1)

# https://stackoverflow.com/questions/55316735/shortest-path-in-a-grid-using-bfs
def find_nearest_bfs(s, grid):
    visited = np.zeros(np.shape(grid))
    # ic(np.shape(visited))
    queue = [(s, [])]  # start point, empty path
    added = set()

    while len(queue) > 0:
        (node, path) = queue.pop(0)
        #ic(node)
        # ic(np.amax(grid))
        # ic(visited[node[0], node[1]])
        path.append(node)
        visited[node[1], node[0]] = 1 #mark visited

        if grid[node[1], node[0]] == 255:
            return path
            break

        adj_nodes = [(node[0], node[1]-1), (node[0], node[1]+1), (node[0]+1, node[1]), (node[0]-1, node[1]), (node[0]+1, node[1]+1), (node[0]+1, node[1]-1), (node[0]-1, node[1]+1), (node[0]-1, node[1]-1)]
        # ic(adj_nodes
        for item in adj_nodes:
            # ic(item)            
            if item[1] >= np.shape(grid)[0] or item[0] >= np.shape(grid)[1]:
                ic("Tried to access array out of bounds")
                ic(item)
                continue
            if visited[item[1], item[0]] == 0 and item not in added: #if not visited
                # ic(item)
                queue.append((item, path[:]))
                added.add(item)
            
        # ic(queue)
        # break

    return path  # no path found

def pathfindBFS(s, e, grid):
    visited = np.zeros(np.shape(grid))
    # ic(np.shape(visited))
    queue = [(s, [])]  # start point, empty path
    added = set()

    while len(queue) > 0:
        (node, path) = queue.pop(0)
        #ic(node)
        # ic(np.amax(grid))
        # ic(visited[node[0], node[1]])
        path.append(node)
        visited[node[1], node[0]] = 1 #mark visited

        if node == e:
            return path
            break

        possible_adj_nodes = [(node[0], node[1]-1), (node[0], node[1]+1), (node[0]+1, node[1]), (node[0]-1, node[1]), (node[0]+1, node[1]+1), (node[0]+1, node[1]-1), (node[0]-1, node[1]+1), (node[0]-1, node[1]-1)]
        adj_nodes = []
        for node in possible_adj_nodes: #only append if we can go there
            if(grid[node[1], node[0]]) == 255:
                adj_nodes.append(node)
        # ic(adj_nodes)
        for item in adj_nodes:
            # ic(item)            
            if visited[item[1], item[0]] == 0 and item not in added: #if not visited
                #ic(item)
                queue.append((item, path[:]))
                added.add(item)
        # ic(queue)
        # break

    return path  # no path found

    
    

#distortion correction

distFrames = 20 #use 20 frames for distortion correction

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

rotTheta = np.zeros(distFrames)

for r in range(distFrames): 

    frame = acqFrame()
    
    dist = cv.undistort(frame,cam,distCoeff)

    # https://github.com/ShaheerSajid/OpenCV-Maze-Solving/blob/main/code/Source.cpp
    hsv = cv.cvtColor(dist, cv.COLOR_BGR2HSV) #src BGR to HSV	
    frame_threshold = cv.inRange(hsv, (0, 0, 228), (255, 255, 255))
    frame_threshold = cv.GaussianBlur(frame_threshold, (9, 9), 1);
    kernel = np.ones((19, 19), np.uint8)
    dilation = cv.dilate(frame_threshold, kernel, iterations=1) 
    erosion = cv.erode(dilation, kernel, iterations=1)
    erosionG = cv.Canny(erosion, 40, 200, None, 3)
    distO = np.copy(dist)    

    rotTheta[r] = 0

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
            cv.imshow("line", distO)
            try:
                rotTheta[r] += (pt1[1]-pt2[1])/(pt1[0]-pt2[0])
                # ic(theta)
            except ZeroDivisionError:
                ic("Rotation Lock Lost")

        rotTheta[r] = rotTheta[r] / len(lines)
        # ic(math.atan(rotTheta[r]))
        # rotTheta[r] = math.atan(rotTheta[r])

rotThetaAvg = np.average(rotTheta)
rotThetaAvg = math.atan(rotThetaAvg)

rotM = cv.getRotationMatrix2D((width/2, height/2), rotThetaAvg * 180 / np.pi, 0.8)
ic(rotM)
# cv.imshow("test",acqCorrectedFrame(distCoeff, rotM, 2, 5))

##cache map
mapFrames = 20 #use 20 frames to get a map and cache it
#targets
redXY = []
greenXY = []
tunnelEndXY = []
path = []

for m in range(mapFrames):

    distOR = acqCorrectedFrame(distCoeff, rotM, 2, 5)    

    corrected = cv.cvtColor(distOR, cv.COLOR_BGR2HSV)
    tresh_path = cv.inRange(corrected, (0, 0, 180), (255, 16, 255))
    tresh_G = cv.inRange(corrected, (66, 30, 70), (85, 255, 255))
    correctedINV = cv.cvtColor(cv.bitwise_not(distOR), cv.COLOR_BGR2HSV)
    # rconv = cv.cvtColor(correctedRGB, cv.COLOR_BGR2LAB)
    # rconv = cv.cvtColor(correctedRGB, cv.COLOR_BGR2YCrCb)
    rconv = correctedINV.copy() #HSV 
    (hr,sr,vr) = cv.split(rconv)
    hr = np.add(hr, 30)
    rconv = cv.merge((hr, sr, vr))
    # tresh_R = cv.inRange(rconv, (110, 108, 101), (183, 186, 139))  #use LAB
    # tresh_R = cv.inRange(rconv, (0, 144, 105), (255, 212, 162)) #use ycbcr
    tresh_R = cv.inRange(rconv, (80, 33, 41), (125, 255, 243)) #use HSV
    tresh_B = cv.inRange(corrected, (low_H, low_S, low_V), (high_H, high_S, high_V))
    tresh_O = cv.inRange(corrected, (27, 91, 182), (52, 255, 255))


    
    # tresh_R = cv.GaussianBlur(tresh_R, (5,5), 1);
    tresh_R = cv.medianBlur(tresh_R, 3)
    rkernel = np.ones((3,3), np.uint8)
    tresh_R = cv.erode(tresh_R, rkernel, iterations = 1)
    tresh_R = cv.dilate(tresh_R, rkernel, iterations=1) 
    # tresh_R = cv.erode(tresh_R, rkernel, iterations=1)

    # tresh_path = cv.GaussianBlur(tresh_path, (15, 15), 1)
    # tresh_path = cv.medianBlur(tresh_path, 3)
    rkernel = np.ones((3,3), np.uint8)    
    tresh_path = cv.dilate(tresh_path, rkernel, iterations=1) 
    tresh_path = cv.erode(tresh_path, rkernel, iterations=1)
    path.append(tresh_path)   

    tresh_G = cv.GaussianBlur(tresh_G, (15,15), 1)
    # tresh_G = cv.medianBlur(tresh_G, 3)
    rkernel = np.ones((3,3), np.uint8)
    tresh_G = cv.erode(tresh_G, rkernel, iterations=1)
    tresh_G = cv.dilate(tresh_G, rkernel, iterations=1)     

    # tresh_O = cv.GaussianBlur(tresh_O, (37, 37), 1);
    tresh_O = cv.medianBlur(tresh_O, 3)
    rkernel = np.ones((15,5), np.uint8) #asymmetrical dilate to make the endpoint detection more reliable
    # tresh_O = cv.erode(tresh_O, rkernel, iterations=1)
    tresh_O = cv.dilate(tresh_O, rkernel, iterations=3) 

    

    # find green square
    cnts,_ = cv.findContours(tresh_G.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    gc = cv.cvtColor(tresh_G, cv.COLOR_GRAY2BGR)
    cv.drawContours(gc, cnts, -1, (0,255,0), 2)
    momentGMax = 0
    idxGMax = 0;
    for i in range(len(cnts)):
        M = cv.moments(cnts[i])
        if M['m00'] > momentGMax:
            idxGMax = i
            momentGMax = M["m00"]
    M = cv.moments(cnts[idxGMax])
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    greenXY.append(np.array([cx, cy]))
    

    #find red square
    cnts,hie = cv.findContours(tresh_R.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    rc = cv.cvtColor(tresh_R, cv.COLOR_GRAY2BGR)
    cv.drawContours(rc, cnts, -1, (0,255,0), 2)
    momentRMax = 0
    idxRMax = 0;
    for i in range(len(cnts)):
        M = cv.moments(cnts[i])
        if M['m00'] > momentRMax:
            idxRMax = i
            momentRMax = M["m00"]
    M = cv.moments(cnts[idxRMax])
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    redXY.append(np.array([cx, cy]))
    
    
    #find tunnel
    cnts,_ = cv.findContours(tresh_O.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    oc = cv.cvtColor(tresh_O, cv.COLOR_GRAY2BGR)
    cv.drawContours(oc, cnts, -1, (0,255,0), 2)

    #find endpoints on path
    
    tresh_pathF = np.float32(tresh_path)
    pathCorner = cv.cornerHarris(tresh_pathF, 5, 3, 0.04)
    pathCorner = cv.erode(pathCorner, None, iterations = 2)
    pathCorner = cv.dilate(pathCorner,None)

    #do temporal filtering
    ntmax = 15    
    if(len(pathCornerStack)) == ntmax:
        pathCornerStack.pop(0) #circular buffer
    pathCornerStack.append(pathCorner)
    pathCornerStackS = np.sum(pathCornerStack, axis = 0)
    pathCornerStackF = np.where(pathCornerStackS>255*12, 255, 0)
    pathCornerStackF = pathCornerStackF.astype(np.uint8)

    # cv.imshow("pathCornerStackF", pathCornerStackF)

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
            if d == 1.0:
                obsEndPoint.append((px, py))
    # obsEndPoint = np.sort(obsEndPoint, axis = -1)
    tunnelEndXY.append(np.array([obsEndPoint[0], obsEndPoint[-1]])) #WARN: This might break stuff
    # tunnelEndXY = np.sort(tunnelEndXY, axis = -1) #sort to remove spurs


ic(tunnelEndXY)

#compute medians
redXY = np.median(redXY, axis=0).astype(int)
greenXY = np.median(greenXY, axis=0).astype(int)
path = np.median(path, axis=0).astype(np.uint8)
tunnelEndXY = np.median(tunnelEndXY, axis=0).astype(int)

ic(tunnelEndXY)

#get path mask
pathMask = np.zeros(np.shape(path))
pathCnts,_ = cv.findContours(path, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
maxPathMoment = 0
maxPath = 0
for i in range(len(pathCnts)): #find largest contour by extents
    currMoment = cv.moments(pathCnts[i])["m00"]
    if currMoment > maxPathMoment:
        maxPathMoment = currMoment
        maxPath = i
pathMask = cv.drawContours(pathMask, pathCnts,i, color=(255,255,255))
pathMask = cv.dilate(pathMask, None, iterations=2)
path = cv.bitwise_and(path, pathMask.astype(np.uint8)) #mask path
 
while True: 

    map = path.copy()
    map = cv.cvtColor(map, cv.COLOR_GRAY2BGR)
    map = drawDiagnosticPoint(map, redXY, (0,0,255))
    map = drawDiagnosticPoint(map, greenXY, (0,255,0))
    map = drawDiagnosticPoint(map, tunnelEndXY[0], (0,255,255))   
    map = drawDiagnosticPoint(map, tunnelEndXY[1], (0,255,255)) 

    #OPENCV USES Y,X!!!!!!! (in the loop)
    targetPos = (256,320) #(x,y)
    # cv.imshow("map", map)
    # ic(path[313,248])
    route = find_nearest_bfs(targetPos, path.copy())
    # ic(route)
    for point in route:
        map = drawDiagnosticPoint(map, point, (255,0,0)) 
    targetEntry = route[-1]
    robotPos = (240,50) 
    route = find_nearest_bfs(robotPos, path.copy())
    map = drawDiagnosticPoint(map, robotPos, (255,0,0)) 
    for point in route:  
        map = drawDiagnosticPoint(map, point, (255,255,0)) 
    robotEntry = route[-1]

    route = pathfindBFS(robotEntry, targetEntry, path.copy())
    for point in route:  
        map = drawDiagnosticPoint(map, point, (3, 20, 220)) 
    # ic(len(route))
    
  
    # Display the resulting frame
    # cv.imshow('frame', path)
    # cv.imshow("red0", rcv[0])
    # cv.imshow("red1", pc)
    # cv.imshow("red2", oc)
    cv.imshow("map", map)
    # end = timer()
    # ic(end-start)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
