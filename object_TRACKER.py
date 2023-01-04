import cv2
from object_detector import ObjectDetector
from kalmanfilter import KalmanFilter
import imutils

# To load the pre recorded video

cam = cv2.VideoCapture("orange.mp4")

#To use the the camera in the device
#cam=cv2.VideoCapture(0)

# Load detector

od = ObjectDetector()

# Load Kalman filter to predict the trajectory
kf = KalmanFilter()

while True:
    ret, frame = cam.read()
    # If the last frame reached break from the loop
    if ret is False:
        break
    # For resizing the frame 
    frame=imutils.resize(frame,height=600)

    
    # To get the coordinates of detected object
    object_bbox = od.detect(frame)
    x, y, x2, y2 = object_bbox

    # To find the centre value
    cx = int((x + x2) / 2)
    cy = int((y + y2) / 2)
    
    #To find the next predicted point
    predicted = kf.predict(cx, cy)
    cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 4)
    cv2.circle(frame, (cx, cy), 10, (255, 0, 255), 5)
    
    cv2.line(frame,(cx,cy),(predicted[0], predicted[1]),(0,255,0),4)
    cv2.circle(frame, (predicted[0], predicted[1]), 10, (0, 0, 255), 7)
    

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(50)
    if key == ord("q"):
        break
