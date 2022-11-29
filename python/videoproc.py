from vidcore import vid
import cv2 as cv
import numpy as np
from icecream import ic
import math



class arena:

    #waypoints
    start = (501, 74)
    redBox = (266, 124)
    greenBox = (741, 130)
    rampStart = (175, 193)
    rampEnd = (190, 621)
    tunnelStart = (841, 242)
    tunnelEnd = (853, 603)
    b1 = (652, 522)
    B1 = (213, 511) #detection (raw coords)
    b2 = (521, 585)
    B2 = (213, 382)
    b3 = (391, 537)
    B3 = (219, 258)
    headingOffset = 180
    locretries = 10

    #config vars
    downscale = 1
    bakRotThetaAvg = 1.5707963267948966
    blockRange = 75 #range within which blocks are considered
    blockRoughnessThres1 = 50000
    blockRoughnessThres2 = 43500
    blockRoughnessThres3 = 43500

    pathLo = (0, 0, 185)
    pathHi = (255, 30, 255)
    pathKernel = (3,3)

    gLo = (66, 30, 70)
    gHi = (85, 255, 255)
    gKernel = (3,3)

    rLo = (100, 33, 41)
    rHi = (120, 255, 243)
    rKernel = (3,3)

    oLo = (27, 50, 182)
    oHi = (52, 255, 255)
    oKernel = (15,5)

    robotMedianBlur = 5
    robotLo = (30, 50, 0)
    robotHi = (50, 255, 255)

    blockLo = (0, 0, 0)
    blockHi = (210, 255, 110)
    blockSmall = 150 #blockSizes
    blockLarge = 400

    ntmax = 15  

    aruco = 119  


    def __init__(self, name):
        self.cap = vid(name)
        ic(self.cap.getRotMAR()) #initialise rotation too
        

    def drawDiagnosticPoint(self, frame, point, color):
        return cv.circle(frame, point, radius=3, color=color, thickness=-1)

    def grabAll(self, nframes = 20):

        img = self.cap.getCorrectedFrame(self.downscale)
        shape = np.shape(cv.cvtColor(img, cv.COLOR_BGR2GRAY))

        self.path = np.empty(shape)
        self.gbox = np.empty(shape)
        self.rbox = np.empty(shape)
        self.obox = np.empty(shape)

        for i in range(nframes):
            img = self.cap.getCorrectedFrame(self.downscale)
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            hsvI = cv.cvtColor(cv.bitwise_not(img), cv.COLOR_BGR2HSV)

            #extract path
            tresh_path = cv.inRange(hsv, self.pathLo, self.pathHi)
            rkernel = np.ones(self.pathKernel, np.uint8)    
            tresh_path = cv.dilate(tresh_path, rkernel, iterations=1) 
            tresh_path = cv.erode(tresh_path, rkernel, iterations=1)
            # tresh_path = cv.GaussianBlur(tresh_path, (5,5), 1)
            # self.path.append(tresh_path) 
            # self.path = np.append(self.path, tresh_path, axis=0) 
            self.path = np.dstack((self.path, tresh_path))
            

            #extract green
            tresh_G = cv.inRange(hsv, self.gLo, self.gHi)
            tresh_G = cv.GaussianBlur(tresh_G, (15,15), 1)
            rkernel = np.ones(self.gKernel, np.uint8)
            tresh_G = cv.erode(tresh_G, rkernel, iterations=1)
            tresh_G = cv.dilate(tresh_G, rkernel, iterations=1)
            self.gbox = np.dstack((self.gbox, tresh_G))
            # ic(self.gbox)

            #extract red
            (hr,sr,vr) = cv.split(hsvI)
            hr = np.add(hr, 30)
            hsvIR = cv.merge((hr, sr, vr))
            tresh_R = cv.inRange(hsvIR, self.rLo, self.rHi)
            tresh_R = cv.medianBlur(tresh_R, 3)
            rkernel = np.ones(self.rKernel, np.uint8)
            tresh_R = cv.erode(tresh_R, rkernel, iterations = 1)
            tresh_R = cv.dilate(tresh_R, rkernel, iterations=1)
            self.rbox = np.dstack((self.rbox, tresh_R))

            #extract lime
            tresh_O = cv.inRange(hsv, self.oLo, self.oHi)
            tresh_O = cv.medianBlur(tresh_O, 3)
            rkernel = np.ones(self.oKernel, np.uint8) #asymmetrical dilate to make the endpoint detection more reliable
            tresh_O = cv.dilate(tresh_O, rkernel, iterations=3) 
            self.obox = np.dstack((self.obox, tresh_O))         


        #compute medians
        self.rbox = np.median(self.rbox, axis=-1).astype(np.uint8)
        self.gbox = np.median(self.gbox, axis=-1).astype(np.uint8)
        self.path = np.median(self.path, axis=-1).astype(np.uint8)
        self.obox = np.median(self.obox, axis=-1).astype(np.uint8)

        # cv.imshow("obox", self.obox)
        # cv.waitKey(500)

        #get path mask
        # self.pathMask = np.zeros(np.shape(self.path))
        # pathCnts,_ = cv.findContours(self.path, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        # maxPathMoment = 0
        # maxPath = 0
        # for i in range(len(pathCnts)): #find largest contour by extents
        #     currMoment = cv.moments(pathCnts[i])["m00"]
        #     if currMoment > maxPathMoment:
        #         maxPathMoment = currMoment
        #         maxPath = i
        # self.pathMask = cv.drawContours(self.pathMask, pathCnts, maxPath, color=(255,255,255))
        # self.pathMask = cv.dilate(self.pathMask, None, iterations=2)
        # self.path = cv.bitwise_and(self.path, self.pathMask.astype(np.uint8)) #mask path

        # find green square
        cnts,_ = cv.findContours(self.gbox, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # gc = cv.cvtColor(tresh_G, cv.COLOR_GRAY2BGR)
        # cv.drawContours(gc, cnts, -1, (0,255,0), 2)
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
        self.gXY=(np.array([cx, cy]))

        #find red square
        cnts,hie = cv.findContours(tresh_R.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # rc = cv.cvtColor(tresh_R, cv.COLOR_GRAY2BGR)
        # cv.drawContours(rc, cnts, -1, (0,255,0), 2)
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
        self.rXY=(np.array([cx, cy]))
        
        #test which are in contour      
        # cv.imshow("obox", self.obox)  
        overlap = cv.bitwise_and(self.obox, self.path)
        overlapRows = np.amax(overlap, axis=1)
        whiteRows = np.where(overlapRows == 255)
        overlapCols = np.amax(overlap, axis = 0)
        whiteCols = np.where(overlapCols == 255)        

        self.t0XY = (whiteCols[0][0], whiteRows[0][0])
        self.t1XY = (whiteCols[0][0], whiteRows[0][-1])

        #extrapolate to find ramp start and end
        rampRow = self.path[whiteRows[0][0]]
        whiteCols = np.where(rampRow == 255)
        self.r0XY = (whiteCols[0][0], whiteRows[0][0])
        rampRow = self.path[whiteRows[0][-1]]
        whiteCols = np.where(rampRow == 255)
        self.r1XY = (whiteCols[0][0], whiteRows[0][-1])

    def findRobot(self):
        img = self.cap.getCorrectedFrame(self.downscale)
        # shape = np.shape(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
        
        mb = cv.medianBlur(img, self.robotMedianBlur)
        mb = cv.cvtColor(mb, cv.COLOR_BGR2HSV)
        robot_tresh = cv.inRange(mb, self.robotLo, self.robotHi)
        robot_tresh = cv.GaussianBlur(robot_tresh, (self.robotMedianBlur, self.robotMedianBlur), 1)
        # cv.imshow("rt", robot_tresh)
        rows = robot_tresh.shape[0]

        circles = cv.HoughCircles(robot_tresh, cv.HOUGH_GRADIENT, 1, rows / 8, param1=10, param2=12, minRadius=5, maxRadius=10)
        ic(circles)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(img, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv.circle(img, center, radius, (255, 0, 255), 3)
        cv.imshow("mb",img)
        if circles is not None:
            return (circles[0][:2]) #return circle center coordinates

    def findRobotAruco(self):
        for i in range(self.locretries):
            data = None
            data = self._findRobotAruco()
            if data is not None:
                (x,y,o) = data
                return(x,y,o)


    def _findRobotAruco(self):
        img = self.cap.getCorrectedFrame(self.downscale)
        h,w,d = img.shape
        hsl = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # thres = cv.inRange(hsl, self.pathLo, self.pathHi)
        thres = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # thres = cv.medianBlur(thres, 3)
        # thres = cv.GaussianBlur(thres, self.pathKernel, 1)
        # cv.imshow("raw", thres)
        # ic("Searching")

        # cv.imshow('raw', img)
        # img = cv.cvtColor(i, cv.COLOR_BGR2GRAY)
        # img

        arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_250)
        arucoParams = cv.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)

        if len(corners) > 0:
        # # flatten the ArUco IDs list
            ids = ids.flatten()
        #     # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
        #         # extract the marker corners (which are always returned in
        #         # top-left, top-right, bottom-right, and bottom-left order)
                corners = markerCorner.reshape((4, 2))
                
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                heading = math.atan2((topLeft[1]-bottomLeft[1]),(topLeft[0]-bottomLeft[0]+1e-6))
        #         # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))
                # heading = math.atan2((topLeft[1]-bottomLeft[1])/(topLeft[0]-bottomLeft[0]+1e-6))
                if heading < 0:
                    heading = np.pi * 2 + heading
                heading = heading * 180 / np.pi
                # heading -= self.headingOffset
            
        		# draw the bounding box of the ArUCo detection
            # cv.line(img, topRight, bottomRight, (0, 255, 0), 2)
            # cv.line(img, topLeft, topRight, (0, 255, 0), 2)
            # cv.line(img, bottomRight, bottomLeft, (0, 255, 0), 2)
            # cv.line(img, bottomLeft, topLeft, (0, 255, 0), 2)
            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            
            # cv.circle(img, (cX, cY), 4, (0, 0, 255), -1)
            # # draw the ArUco marker ID on the image
            # cv.putText(img, str(markerID),
            #     (topLeft[0], topLeft[1] - 15), cv.FONT_HERSHEY_SIMPLEX,
            #     0.5, (0, 255, 0), 2)
            # print("[INFO] ArUco marker ID: {}".format(markerID))
            # show the output image
            # cv.imshow("Image", img)
            # cv.waitKey(0)
            ic(cX, cY, heading)
            return (cX,cY, heading)
            

    

    def detectBlocks(self):
        # img = self.cap.getCorrectedFrame(self.downscale)
        imger = []
        for n in range(100): #median blur to get better resolution
            img = self.cap.getRaw()
            imger.append(img)
        # ic(np.shape(imger))
        img = np.mean(imger, axis = 0).astype(np.uint8)
        # ic(np.shape(img))

        
        b,g,r = cv.split(img)
        luv = cv.cvtColor(img, cv.COLOR_BGR2LUV)
        l,u,v = cv.split(luv)

        thres = cv.inRange(luv,(0, 0, 0),(210, 255, 110))
        thresint = cv.cvtColor(thres, cv.COLOR_GRAY2BGR)
        
        # cv.imshow('l', l)
        # cv.imshow('u', u)
        # cv.imshow('v', v)

        cnts, _ = cv.findContours(thres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(thresint, cnts, -1, (0,255,0), 2)
        blocks = []
        for c in cnts:
            M = cv.moments(c)
            # ic(M['m00'])
            if M["m00"] < self.blockLarge and M["m00"] > self.blockSmall:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                #start FFT magic
                x,y,w,h = cv.boundingRect(c) #get bounding box
                croppedThres = thres[y: y+h, x: x+h]
                croppedR = r[y: y+h, x: x+h]
                # cv.imshow("r", croppedR)
                # cv.imshow("cthres", croppedThres)
                thresBlur = cv.GaussianBlur(croppedThres, (5,5), 0, 0)
                thresBlur = thresBlur.astype('float64')
                thresMax = np.max(abs(thresBlur))
                thresBlur *= 1/thresMax
                croppedR = croppedR.astype('float64')
                croppedR *= thresBlur
                croppedR = croppedR.astype('uint8')
                det = cv.equalizeHist(croppedR) 
                cv.imshow("det", det)          
                laplacian = cv.Laplacian(det, cv.CV_64F)
                sobel = cv.Sobel(det, cv.CV_64F, 1, 1)
                masked = croppedThres * sobel
                # cv.imshow("crop", masked)
                f = np.fft.fft2(det)

                #LPF Magic
                cutoff = (np.divide(masked.shape, 5).astype('int'))#discard lower freqs
                fmask = np.zeros(f.shape)
                for y in range(cutoff[0], fmask.shape[0]- cutoff[0]+1):
                    for x in range(cutoff[1], fmask.shape[1] - cutoff[1]+1):
                        fmask[y, x] = 1
                ff = np.multiply(f, fmask)
                roughness = np.sum(np.abs(ff))

                blocks.append((cx, cy, roughness))

        ic(blocks)

        blockIden = {
            "b1": None,
            "b2": None,
            "b3": None,
        }

        for block in blocks:
            if math.dist(block[:2], self.B1) < self.blockRange:
                if block[2] > self.blockRoughnessThres1:
                    blockIden['b1'] = "rough"
                else:
                    blockIden['b1'] = "smooth"
            elif math.dist(block[:2], self.B2) < self.blockRange:
                if block[2] > self.blockRoughnessThres2:
                    blockIden['b2'] = "rough"
                else:
                    blockIden['b2'] = "smooth"
            elif math.dist(block[:2], self.B3) < self.blockRange:
                if block[2] > self.blockRoughnessThres3:
                    blockIden['b3'] = "rough"
                else:
                    blockIden['b3'] = "smooth"


        # cv.imshow("ti", thresint)
        
        return blockIden

    def identifyNearestBlock(self):
        pass
        


    def getDiagnosticMap(self):
        self.grabAll()
        self.map = cv.cvtColor(self.path, cv.COLOR_GRAY2BGR)
        self.map = self.drawDiagnosticPoint(self.map, self.rXY, (0,0,255))
        self.map = self.drawDiagnosticPoint(self.map, self.gXY, (0,255,0))
        self.map = self.drawDiagnosticPoint(self.map, self.t0XY, (0,255,255))   
        self.map = self.drawDiagnosticPoint(self.map, self.t1XY, (0,255,255))
        self.map = self.drawDiagnosticPoint(self.map, self.r1XY, (255,0,255))   
        self.map = self.drawDiagnosticPoint(self.map, self.r0XY, (255,0,255))
        blocks = self.detectBlocks()
        if(len(blocks) > 0):
            for b in blocks:
                self.map = self.drawDiagnosticPoint(self.map, b, (255,0,0))
        # self.map = self.drawDiagnosticPoint(self.map, self.findRobot(), (127,255,255))
        return self.map

        


if __name__ == "__main__":
    map = arena("http://localhost:8081/stream/video.mjpeg")
    # img = map.getDiagnosticMap()
    # map = arena("C:\\Users\\yehen\\Videos\\2022-11-10 09-09-04.m4v")
    while True:
        
        ic(map.findRobotAruco())
        # ic(map.detectBlocks())
        # img = map.getDiagnosticMap()

        # cv.imshow("test", img)
        if cv.waitKey(1) == ord('q'):
            break

    cv.destroyAllWindows()
