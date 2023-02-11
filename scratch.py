import numpy
import cv2


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
while True:
    ret,img = cap.read()
    if not ret:
        raise RuntimeError("failed to read frame")
    img = cv2.flip(img,1)
    cv2.imshow('img',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break