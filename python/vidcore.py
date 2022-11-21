from threadedCapture import VideoCapture
import numpy as np
import cv2 as cv
import math
from icecream import ic




# def getUndistort(cap: VideoCapture, balance=0.0, dim2=None, dim3=None):
#     img = cap.read()
#     dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort    
#     if not dim2:
#         dim2 = dim1    
#     if not dim3:
#         dim3 = dim1    
#         scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
#     scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
#     new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
#     map1, map2 = cv.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv.CV_16SC2)
#     undistorted_img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT) 

#     return undistorted_img


# def getRotM(cap: VideoCapture, nframes: int = 20):

#     (height, width ,dim) = np.shape(getUndistort(cap))
#     # ic(width)
    
#     rotTheta = np.zeros(nframes)
#     for r in range(nframes):
#         dist = getUndistort(cap)
#         hsv = cv.cvtColor(dist, cv.COLOR_BGR2HSV) #src BGR to HSV	
#     frame_threshold = cv.inRange(hsv, (0, 0, 228), (255, 255, 255))
#     frame_threshold = cv.GaussianBlur(frame_threshold, (9, 9), 1);
#     kernel = np.ones((19, 19), np.uint8)
#     dilation = cv.dilate(frame_threshold, kernel, iterations=1) 
#     erosion = cv.erode(dilation, kernel, iterations=1)
#     erosionG = cv.Canny(erosion, 40, 200, None, 3)
#     distO = np.copy(dist)    

#     rotTheta[r] = 0

#     lines = cv.HoughLines(erosionG, 1, np.pi / 250, 250, None, 0, 0)
#     # ic(lines)
#     if lines is not None:
#         for i in range(0, len(lines)):
#             rho = lines[i][0][0]
#             theta = lines[i][0][1]
#             a = math.cos(theta)
#             b = math.sin(theta)
#             x0 = a * rho
#             y0 = b * rho
#             pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#             pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#             cv.line(distO, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
#             # cv.imshow("line", distO)
#             try:
#                 rotTheta[r] += (pt1[1]-pt2[1])/(pt1[0]-pt2[0])
#                 # ic(theta)
#             except ZeroDivisionError:
#                 ic("Zero in Line Detection")

#         rotTheta[r] = rotTheta[r] / len(lines)
    
#     rotThetaAvg = np.average(rotTheta)
#     rotThetaAvg = math.atan(rotThetaAvg)
#     rotThetaAvg += np.pi/2
#     rotM = cv.getRotationMatrix2D((width/2, height/2), rotThetaAvg * 180 / np.pi, 1) #???????

#     # return rotThetaAvg
#     return rotM


# def getCorrectedFrame(cap: VideoCapture, rotM, resizeFactor = 1, blurSize = 1):

#     (_height, _width,_dim) = np.shape(getUndistort(cap))
#     dist = getUndistort(cap)
    

#     distOR = cv.warpAffine(src=dist, M=rotM, dsize=(_width, _height))
#     dim = (int(_width/resizeFactor), int(_height/resizeFactor))
#     distOR = cv.GaussianBlur(distOR, (blurSize, blurSize), 1)
#     distOR = cv.resize(distOR, dim, interpolation = cv.INTER_AREA)

#     return distOR

class vid:
    DIM=(1012, 760)
    K = np.array([[567.4130565572482, 0.0, 501.39791714355], [0.0, 567.3325405728447, 412.9039077874256], [0.0, 0.0, 1.0]])
    D = np.array([[-0.05470334257497442], [-0.09142371384400942], [0.17966906821072895], [-0.08708720575337928]])
    
    def __init__(self, name) -> None:
        # self.cap = VideoCapture(name)
        self.cap = cv.VideoCapture(name)
        pass

    def getUndistort(self, balance=0.0, dim2=None, dim3=None):
        ret, img = self.cap.read()
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

    def getRotM(self, nframes: int = 20):

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

        lines = cv.HoughLines(erosionG, 1, np.pi / 250, 250, None, 0, 0)
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
        
        rotThetaAvg = np.average(rotTheta)
        rotThetaAvg = math.atan(rotThetaAvg)
        rotThetaAvg += np.pi/2
        self.rotM = cv.getRotationMatrix2D((width/2, height/2), rotThetaAvg * 180 / np.pi, 1) #???????

        # return rotThetaAvg
        return self.rotM

    def getCorrectedFrame(self, resizeFactor = 1, blurSize = 1):

        (_height, _width,_dim) = np.shape(self.getUndistort(self.cap))
        dist = self.getUndistort(self.cap)
        

        distOR = cv.warpAffine(src=dist, M=self.rotM, dsize=(_width, _height))
        dim = (int(_width/resizeFactor), int(_height/resizeFactor))
        distOR = cv.GaussianBlur(distOR, (blurSize, blurSize), 1)
        distOR = cv.resize(distOR, dim, interpolation = cv.INTER_AREA)

        return distOR




if __name__ == "__main__":

    # cap = vid("http://localhost:8081/stream/video.mjpeg")
    
    cap = vid("C:\\Users\\yehen\\Videos\\2022-11-10 09-09-04.m4v")
    cap.getRotM() #enable rotation

    while True:
        img = cap.getCorrectedFrame(2)
        # ic(np.shape(img))
        cv.imshow("test", img)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()