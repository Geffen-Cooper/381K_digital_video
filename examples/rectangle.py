import cv2
import time
import numpy as np

# init rect
RECT_W = 100
RECT_H = 100
RECT_Vx = 10
RECT_Vy = 10

# initialize the capture object
cap = cv2.VideoCapture(0)
FRAME_W = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
FRAME_H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 

# create a window
cv2.namedWindow("window", cv2.WINDOW_NORMAL)

# measure FPS
last_frame_time = 0
curr_frame_time = 0
frame_count = 0
fps = 0
font = cv2.FONT_HERSHEY_SIMPLEX

# start rectangle at the center
center_tlc_x = int(FRAME_W/2 - RECT_W/2)
center_tlc_y = int(FRAME_H/2 - RECT_H/2)
center_brc_x = int(FRAME_W/2 + RECT_W/2)
center_brc_y = int(FRAME_H/2 + RECT_H/2)
start_point = np.array([center_tlc_x, center_tlc_y])
end_point = np.array([center_brc_x, center_brc_y])


# start loop
while True:
    # try to read a frame
    ret,img = cap.read()
    if not ret:
        raise RuntimeError("failed to read frame")

    # flip horizontally
    img = cv2.flip(img,1)

    # === Rectangle motion update ===
    # update position
    start_point[0] += RECT_Vx
    start_point[1] += RECT_Vy
    end_point[0] += RECT_Vx
    end_point[1] += RECT_Vy
    # check collision
    if start_point[0] <= 0: # left
        RECT_Vx *= -1
    if start_point[1] <= 0: # top
        RECT_Vy *= -1
    if end_point[0] >= FRAME_W: # right
        RECT_Vx *= -1
    if end_point[1] >= FRAME_H: # bottom
        RECT_Vy *= -1
   
    cv2.rectangle(img, start_point, end_point, (0,0,255), 4)
  
    # calculate the fps
    frame_count += 1
    curr_frame_time = time.time()
    diff = curr_frame_time - last_frame_time
    if diff > 1:
        fps = str(round(frame_count/(diff),2))
        last_frame_time = curr_frame_time
        frame_count = 0

    # img, text, location of BLC, font, size, color, thickness, linetype
    cv2.putText(img, fps+", "+str(int(FRAME_W))+"x"+str(int(FRAME_H)), (7, 30), font, 1, (100, 255, 0), 1, cv2.LINE_AA)
    
    cv2.imshow('window',img)
    # quit when click 'q' on keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break