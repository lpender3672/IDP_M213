import cv2 as cv
import numpy as np

cap = cv.VideoCapture("http://localhost:8081/stream/video.mjpeg")


width  = 1012
height = 760

distCoeff = np.zeros((4,1),np.float64)

# TODO: add your coefficients here!
k1 = -2.5e-5; # negative to remove barrel distortion
k2 = 0.0;
p1 = 0.0;
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

# cap = cv.undistort(cap,cam,distCoeff)



while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    gray = cv.undistort(gray,cam,distCoeff)

    # Display the resulting frame
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
