import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

DEBUG_MODE = False

# init rect
RECT_W = 100
RECT_H = 100
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


sx, sy = np.meshgrid(np.arange(-15,15+1),np.arange(-15,15+1))

# start loop
while True:
    # try to read a frame
    ret,img = cap.read()
    if not ret:
        raise RuntimeError("failed to read frame")

    # flip horizontally
    img = cv2.flip(img,1)
    curr_rect_px = img[start_point[1]:end_point[1],start_point[0]:end_point[0]]

    # === Rectangle tracking update ===
    # update position
    if start_tracking:
        # l1 = np.ones(9)*1e9
        l1 = np.ones((sy.shape[0],sx.shape[1]))*1e9
        for i,(xr,yr) in enumerate(zip(sx,sy)):
            # get the columns of xr and yr
            for j,(xc,yc) in enumerate(zip(xr,yr)):
                if start_point[1]+yc <= 0 or end_point[1]+yc >= FRAME_H or start_point[0]+xc <= 0 or end_point[0]+xc >= FRAME_W:
                    print("Hitting Boundary")
                else:
                    pred_px = img[start_point[1]+yc:end_point[1]+yc,start_point[0]+xc:end_point[0]+xc]
                    l1[i,j] = np.sum(np.abs(pred_px.astype(int)-last_rect_px.astype(int)))
                # # get the adjacent pixels absolute difference with last square
                # if start_point[1]+dir[1]*RECT_W/factor <= 0 or end_point[1]+dir[1]*RECT_W/factor >= FRAME_H or start_point[0]+dir[0]*RECT_W/factor <= 0 or end_point[0]+dir[0]*RECT_W/factor >= FRAME_W:
                #     print("BAD")
                # else:
                #     # print(start_point[1],dir[1]*RECT_W/2)
                #     # print(start_point[1]+dir[1]*RECT_W/2,end_point[1]+dir[1]*RECT_W/2,start_point[0]+dir[0]*RECT_W/2,end_point[0]+dir[0]*RECT_W/2)
                #     pred_px = img[start_point[1]+dir[1]*RECT_W//factor:end_point[1]+dir[1]*RECT_W//factor,start_point[0]+dir[0]*RECT_W//factor:end_point[0]+dir[0]*RECT_W//factor]
                #     l1[i] = np.sum(np.abs(pred_px.astype(int)-last_rect_px.astype(int)))
                #     if DEBUG_MODE:
                #         cv2.imwrite("patch"+str(dir)+".png",pred_px)
        if DEBUG_MODE:
            for idx in np.argsort(l1):
                print(dirs[idx],l1[idx])
            print("===")
            show_patches()
        closest = np.argmin(l1)
        row = closest % 31
        col = closest //31
        # print(row,col)
        # print(dirs[closest])
        start_point[0] += row-15#dirs[closest][0]*RECT_W/factor
        start_point[1] += col-15#dirs[closest][1]*RECT_W/factor
        end_point[0] += row-15#dirs[closest][0]*RECT_W/factor
        end_point[1] += col-15#dirs[closest][1]*RECT_W/factor
    else:
        min_noise = np.sum(np.abs(curr_rect_px-last_rect_px))

    # start_point[0] += RECT_Vx
    # start_point[1] += RECT_Vy
    # end_point[0] += RECT_Vx
    # end_point[1] += RECT_Vy
    # check collision
    if start_point[0] <= 0: # left
        start_point[0] = 0
        end_point[0] = RECT_W
    if start_point[1] <= 0: # top
        start_point[1] = 0
        end_point[1] = RECT_H
    if end_point[0] >= FRAME_W: # right
        end_point[0] = FRAME_W
        start_point[0] = FRAME_W - RECT_W
    if end_point[1] >= FRAME_H: # bottom
        end_point[1] = FRAME_H
        start_point[1] = FRAME_H - RECT_H
   
    last_rect_px = img[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    cv2.imwrite("last.png",last_rect_px)
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

    # get key
    k = cv2.waitKey(1)

    if k == ord('q'):
        cv2.destroyAllWindows()
        break
    elif k == ord('s'):
        start_tracking = True
