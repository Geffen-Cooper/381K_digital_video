import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

DEBUG_MODE = False

# init rect
RECT_W = 128
RECT_H = 128
# RECT_Vx = 10
# RECT_Vy = 10
factor = 20

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

start_tracking = False

# try to read a frame
ret,img = cap.read()
if not ret:
    raise RuntimeError("failed to read frame")

# flip horizontally
img = cv2.flip(img,1)
last_rect_px = img[start_point[1]:end_point[1],start_point[0]:end_point[0]]

SEARCH_SIZE = 15
sx, sy = np.meshgrid(np.arange(-SEARCH_SIZE,SEARCH_SIZE+1),np.arange(-SEARCH_SIZE,SEARCH_SIZE+1))

x_rel = 0
y_rel = 0

# pad to expected output size
center_patch = np.zeros((RECT_H*2+RECT_H-1,RECT_W*2+RECT_W-1), np.uint8)
search_patch = np.zeros((RECT_H*2+RECT_H-1,RECT_W*2+RECT_W-1), np.uint8)
# filter = np.zeros((int(FRAME_H+3-1),int(FRAME_W+3-1)), np.uint8)
# input = np.zeros((int(FRAME_H+3-1),int(FRAME_W+3-1)), np.uint8)
# sobel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]],np.uint8)
# start loop
while True:
    # try to read a frame
    ret,img = cap.read()
    if not ret:
        raise RuntimeError("failed to read frame")

    # flip horizontally
    img = cv2.flip(img,1)
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_ = cv2.GaussianBlur(img_,(11,11),7,7)
    
    # === Rectangle tracking update ===
    # update position
    if start_tracking:
        # center_patch = img[start_point[1]:end_point[1],start_point[0]:end_point[0]]
        center_patch[RECT_H:RECT_H*2,RECT_W:RECT_W*2] = img_[start_point[1]:end_point[1],start_point[0]:end_point[0],0]
        center_patch = cv2.flip(center_patch,0)
        center_patch = cv2.flip(center_patch,1)
        # filter[int(FRAME_H//2-1-1):int(FRAME_H//2-1-1+3),int(FRAME_W//2-1-1):int(FRAME_W//2-1-1+3)] = sobel
        # input[1:int(1+FRAME_H),1:int(1+FRAME_W)] = img_[:,:,0]
        search_patch[RECT_H-RECT_H//2:RECT_H*2+RECT_H//2,RECT_W-RECT_W//2:RECT_W*2+RECT_W//2] = img_[start_point[1]-RECT_H//2:end_point[1]+RECT_H//2,start_point[0]-RECT_W//2:end_point[0]+RECT_W//2,0]
        
        F1 = np.fft.fft2(center_patch)
        F2 = np.fft.fft2(search_patch)
        F3 = F1*F2
        _img = np.fft.ifft2(F3).astype(np.uint8)
        _img = np.fft.fftshift(_img)
        closest = np.argmax(_img)
        col = (closest % (RECT_H+RECT_W))
        row = (closest // (RECT_H+RECT_W))
        _img[row,:] = 255
        _img[:,col] = 255

        cv2.imshow('patch',img_[:,:,0])
    cv2.rectangle(img, start_point, end_point, (0,0,255), 4)
    cv2.circle(img, (start_point[0]+RECT_W//2,start_point[1]+RECT_H//2), radius=6, color=(0, 0, 255), thickness=-1)
    # center_patch *= 0
    # search_patch *= 0

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

    # get key
    k = cv2.waitKey(1)

    if k == ord('q'):
        cv2.destroyAllWindows()
        break
    elif k == ord('s'):
        start_tracking = True
