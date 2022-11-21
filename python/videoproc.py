from vidcore import vid
import cv2 as cv
import numpy as np
from icecream import ic



class arena:

    #config vars
    downscale = 2

    pathLo = (0, 0, 180)
    pathHi = (255, 16, 255)
    pathKernel = (3,3)

    gLo = (66, 30, 70)
    gHi = (85, 255, 255)
    gKernel = (3,3)

    rLo = (80, 33, 41)
    rHi = (125, 255, 243)
    rKernel = (3,3)

    oLo = (27, 91, 182)
    oHi = (52, 255, 255)
    oKernel = (15,5)

    ntmax = 15 

    
    


    def __init__(self, name):
        self.cap = vid(name)
        ic(self.cap.getRotM()) #initialise rotation too
        self.redXY = []
        self.greenXY = []
        self.tunnelEndXY = []
        self.path = []
        self.pathCornerStack = []

    def drawDiagnosticPoint(self, frame, point, color):
        return cv.circle(frame, point, radius=1, color=color, thickness=-1)

    def grabAll(self, nframes = 20):

        img = self.cap.getCorrectedFrame(self.downscale)
        shape = np.shape(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
        # self.redXY = np.empty(shape)
        # self.greenXY = np.empty(shape)
        # self.tunnelEndXY = np.empty(shape)
        self.path = np.empty(shape)
        self.pathCornerStack = np.empty(shape)

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
            

            #extract green
            tresh_G = cv.inRange(hsv, self.gLo, self.gHi)
            tresh_G = cv.GaussianBlur(tresh_G, (15,15), 1)
            # tresh_G = cv.medianBlur(tresh_G, 3)
            rkernel = np.ones(self.gKernel, np.uint8)
            tresh_G = cv.erode(tresh_G, rkernel, iterations=1)
            tresh_G = cv.dilate(tresh_G, rkernel, iterations=1)

            #extract red
            (hr,sr,vr) = cv.split(hsvI)
            hr = np.add(hr, 30)
            hsvIR = cv.merge((hr, sr, vr))
            tresh_R = cv.inRange(hsvIR, self.rLo, self.rHi)
            tresh_R = cv.medianBlur(tresh_R, 3)
            rkernel = np.ones(self.rKernel, np.uint8)
            tresh_R = cv.erode(tresh_R, rkernel, iterations = 1)
            tresh_R = cv.dilate(tresh_R, rkernel, iterations=1) 

            #extract lime
            tresh_O = cv.inRange(hsv, self.oLo, self.oHi)
            tresh_O = cv.medianBlur(tresh_O, 3)
            rkernel = np.ones(self.oKernel, np.uint8) #asymmetrical dilate to make the endpoint detection more reliable
            tresh_O = cv.dilate(tresh_O, rkernel, iterations=3) 

            # find green square
            cnts,_ = cv.findContours(tresh_G.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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
            self.greenXY.append(np.array([cx, cy]))

            
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
            self.redXY.append(np.array([cx, cy]))
            
            
            #find tunnel
            cnts,_ = cv.findContours(tresh_O.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # oc = cv.cvtColor(tresh_O, cv.COLOR_GRAY2BGR)
            # cv.drawContours(oc, cnts, -1, (0,255,0), 2)

            #find endpoints on path            
            tresh_pathF = np.float3q2(tresh_path)
            pathCorner = cv.cornerHarris(tresh_pathF, 1, 9, 0.01)
            # pathCorner = cv.erode(pathCorner, None, iterations = 3)
            # pathCorner = cv.dilate(pathCorner,None, iterations=5)
            pathCorner = pathCorner.astype(np.uint8)
            cv.imshow("treshpath", tresh_pathF)
            # ic(pathCorner.shape)
            

            # if(len(self.pathCornerStack)) == self.ntmax:
            #     self.pathCornerStack.pop(0) #circular buffer
            # self.pathCornerStack = np.append(self.pathCornerStack, pathCorner, axis = 0)
            # self.pathCornerStackS = np.sum(self.pathCornerStack, axis = 0)
            # self.pathCornerStackF = np.where(self.pathCornerStackS>255*12, 255, 0)
            # self.pathCornerStackF = self.pathCornerStackF.astype(np.uint8)

            #test which are in contour
            tresh_OD = cv.dilate(tresh_O, None, iterations = 5)
            cntsO,_ = cv.findContours(tresh_OD.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cntsP,_ = cv.findContours(pathCorner.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # pc = cv.cvtColor(pathCorner, cv.COLOR_GRAY2BGR)
            # oc = cv.cvtColor(tresh_OD, cv.COLOR_GRAY2BGR)
            # cv.drawContours(pc, cntsP, -1, (0,255,0), 1)
            cv.imshow("test", pathCorner)
            cv.waitKey(10000)
            obsEndPoint = []
            for conO in cntsO:
                for conP in cntsP:
                    mu = cv.moments(conP)
                    (px, py) = (int(mu['m10'] / (mu['m00'] + 1e-5)), int(mu['m01'] / (mu['m00'] + 1e-5)))
                    d = cv.pointPolygonTest(conO, (px, py), False)
                    if d == 1.0:
                        obsEndPoint.append((px, py))
            # obsEndPoint = np.sort(obsEndPoint, axis = -1)
            # ic(obsEndPoint)
            self.tunnelEndXY.append(np.array([obsEndPoint[0], obsEndPoint[-1]])) #WARN: This might break stuff

        self.redXY = np.median(self.redXY, axis=0).astype(int)
        self.greenXY = np.median(self.greenXY, axis=0).astype(int)
        self.path = np.median(self.path, axis=0).astype(np.uint8)
        self.tunnelEndXY = np.median(self.tunnelEndXY, axis=0).astype(int)

        #get path mask
        self.pathMask = np.zeros(np.shape(self.path))
        pathCnts,_ = cv.findContours(self.path, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        maxPathMoment = 0
        maxPath = 0
        for i in range(len(pathCnts)): #find largest contour by extents
            currMoment = cv.moments(pathCnts[i])["m00"]
            if currMoment > maxPathMoment:
                maxPathMoment = currMoment
                maxPath = i
        self.pathMask = cv.drawContours(self.pathMask, pathCnts, maxPath, color=(255,255,255))
        self.pathMask = cv.dilate(self.pathMask, None, iterations=2)
        self.path = cv.bitwise_and(self.path, self.pathMask.astype(np.uint8)) #mask path

    def getDiagnosticMap(self):
        self.grabAll()
        self.map = self.path.copy()
        # self.map = cv.cvtColor(map, cv.COLOR_GRAY2BGR)
        self.map = self.drawDiagnosticPoint(self.map, self.redXY, (0,0,255))
        self.map = self.drawDiagnosticPoint(self.map, self.greenXY, (0,255,0))
        self.map = self.drawDiagnosticPoint(self.map, self.tunnelEndXY[0], (0,255,255))   
        self.map = self.drawDiagnosticPoint(self.map, self.tunnelEndXY[1], (0,255,255))
        return self.map

        


if __name__ == "__main__":
    map = arena("http://localhost:8081/stream/video.mjpeg")
    while True:
        img = map.getDiagnosticMap()

        cv.imshow("test", img)
        if cv.waitKey(1) == ord('q'):
            break

    cv.destroyAllWindows()
