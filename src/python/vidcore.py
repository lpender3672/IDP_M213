from threadedCapture import VideoCapture
import numpy as np
import cv2 as cv
import math
from icecream import ic

class vid:
    #distortion correction coefficients
    DIM=(1012, 760)
    K = np.array([[567.4130565572482, 0.0, 501.39791714355], [0.0, 567.3325405728447, 412.9039077874256], [0.0, 0.0, 1.0]])
    D = np.array([[-0.05470334257497442], [-0.09142371384400942], [0.17966906821072895], [-0.08708720575337928]])
    
    def __init__(self, name) -> None:
        self.cap = VideoCapture(name)
        pass

    #get uncorrected frame
    def getRaw(self):
        img = self.cap.read()
        return img

    #get undistorted frame
    def getUndistort(self, balance=0.0, dim2=None, dim3=None):
        img = self.cap.read()
        dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort    
        if not dim2:
            dim2 = dim1    
        if not dim3:
            dim3 = dim1    
            scaled_K = self.K * dim1[0] / self.DIM[0]  # The values of K is to scale with image dimension.
        scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
        new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, self.D, dim2, np.eye(3), balance=0)
        map1, map2 = cv.fisheye.initUndistortRectifyMap(scaled_K, self.D, np.eye(3), new_K, dim3, cv.CV_16SC2)
        undistorted_img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT) 

        return undistorted_img

    #automatic rotation correction with Hough Lines transform
    def getRotMHL(self, nframes: int = 20):

        (height, width ,dim) = np.shape(self.getUndistort(self.cap))
        # ic(width)
        
        rotTheta = np.zeros(nframes)
        for r in range(nframes):
            dist = self.getUndistort(self.cap)
            hsv = cv.cvtColor(dist, cv.COLOR_BGR2HSV) #src BGR to HSV	
        frame_threshold = cv.inRange(hsv, (0, 0, 228), (255, 255, 255))
        frame_threshold = cv.GaussianBlur(frame_threshold, (9, 9), 1);
        kernel = np.ones((19, 19), np.uint8)
        dilation = cv.dilate(frame_threshold, kernel, iterations=1) 
        erosion = cv.erode(dilation, kernel, iterations=1)
        erosionG = cv.Canny(erosion, 40, 200, None, 3)
        distO = np.copy(dist)    

        rotTheta[r] = 0

        lines = cv.HoughLines(erosionG, 1, np.pi / 250, 220, None, 0, 0)
        # ic(lines)
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
                # cv.imshow("line", distO)
                try:
                    rotTheta[r] += (pt1[1]-pt2[1])/(pt1[0]-pt2[0])
                    # ic(theta)
                except ZeroDivisionError:
                    ic("Zero in Line Detection")

            rotTheta[r] = rotTheta[r] / len(lines)
        
        #calculate how much the frame is rotated and reverse that rotation
        rotThetaAvg = np.average(rotTheta)
        rotThetaAvg = math.atan(rotThetaAvg)
        rotThetaAvg += np.pi/2
        # ic(rotThetaAvg)
        self.rotM = cv.getRotationMatrix2D((width/2, height/2), rotThetaAvg * 180 / np.pi, 1)

        # return rotThetaAvg
        return self.rotM

    #WARN: Broken with blocks present
    def getRotMMAR(self): #min area rect implementation of automatic rotation correction
        (height, width ,dim) = np.shape(self.getUndistort(self.cap))
        pathLo = (0, 0, 180)
        pathHi = (255, 30, 255)
        img = self.getUndistort(self.cap)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        tresh_path = cv.inRange(hsv,pathLo, pathHi)
        # rkernel = np.ones(self.pathKernel, np.uint8)    
        
        tresh_path = cv.erode(tresh_path, (5,5), iterations=3)
        tresh_path = cv.dilate(tresh_path, (3,3), iterations=3) 

        # cv.imshow('thres', tresh_path)

        cnt, _ = cv.findContours(tresh_path, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnt = sorted(cnt, key = lambda x: cv.arcLength(curve = x, closed=False), reverse=True) #get largest contour
        cnt = cnt[0]
        rect = cv.minAreaRect(cnt)
        # box = cv.boxPoints(rect)
        # box = np.int0(box)
        # cv.drawContours(img,[box],0,(0,0,255),2)
        # cv.drawContours(cnt,[box],0,(0,0,255),2)
        # cv.imshow("box", img)
        # cv.waitKey(1000000)
        # ic(rect)
        self.rotM = cv.getRotationMatrix2D((width/2, height/2), rect[2], 1)
        return self.rotM

    #precalibrated fallback value of rotation angle
    def getRotMAR(self):
        (height, width ,dim) = np.shape(self.getUndistort(self.cap))

        self.rotM = cv.getRotationMatrix2D((width/2, height/2), 1.5707963267948966 * 180 /np.pi, 1)
        return self.rotM

    #return undistorted and straightened frame
    def getCorrectedFrame(self, resizeFactor = 1, blurSize = 1):

        (_height, _width,_dim) = np.shape(self.getUndistort(self.cap))
        dist = self.getUndistort(self.cap)        

        #resize
        distOR = cv.warpAffine(src=dist, M=self.rotM, dsize=(_width, _height))
        dim = (int(_width/resizeFactor), int(_height/resizeFactor))
        distOR = cv.GaussianBlur(distOR, (blurSize, blurSize), 1)
        distOR = cv.resize(distOR, dim, interpolation = cv.INTER_AREA)

        return distOR


#this section only for testing

if __name__ == "__main__":

    
    cap = vid("http://localhost:8081/stream/video.mjpeg")
    # cap = vid("C:\\Users\\yehen\\Videos\\2022-11-10 09-09-04.m4v")
    cap.getRotMAR() #enable rotation

    while True:
        img = cap.getCorrectedFrame(2)
        # ic(np.shape(img))
        cv.imshow("test", img)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()