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
factor = 10

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

# sx, sy = np.meshgrid(np.arange(-start_point[0],start_point[0]+1),np.arange(-start_point[1],start_point[1]+1))
sx, sy = np.meshgrid(np.arange(-50,50+1),np.arange(-50,50+1))

start_tracking = False

# try to read a frame
ret,img = cap.read()
if not ret:
    raise RuntimeError("failed to read frame")

# flip horizontally
img = cv2.flip(img,1)
last_rect_px = img[start_point[1]:end_point[1],start_point[0]:end_point[0]]

l1 = np.ones(9)*1e10
dirs = np.array([[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]])

def show_patches():
    fig,axs = plt.subplot_mosaic([['last','[-1 -1]','[ 0 -1]','[ 1 -1]'],
                                   ['last','[-1  0]','[0 0]','[1 0]'],
                                   ['last','[-1  1]','[0 1]','[1 1]']],figsize=(6,6))
    for dir in dirs:
        img = cv2.imread("patch"+str(dir)+".png")
        axs[str(dir)].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = cv2.imread("last.png")
    axs['last'].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

# start loop
while True:
    # try to read a frame
    ret,img = cap.read()
    if not ret:
        raise RuntimeError("failed to read frame")

    # flip horizontally
    img = cv2.flip(img,1)

    # === Rectangle tracking update ===
    # update position
    if start_tracking:
        l1 = np.zeros((sy.shape[0],sx.shape[1]))
        print(l1.shape)
        # get the rows of sx and sy
        for i,(xr,yr) in enumerate(zip(sx,sy)):
            # get the columns of xr and yr
            for j,(xc,yc) in enumerate(zip(xr,yr)):
                try:
                    pred_px = img[start_point[1]+yc:end_point[1]+yc,start_point[0]+xc:end_point[0]+xc]
                    l1[i,j] = np.sum(np.abs(pred_px.astype(int)-last_rect_px.astype(int)))
                except:
                    print(start_point,end_point)
                    print(i,j,xc,yc)
                    print(pred_px.shape)
                    print(last_rect_px.shape)
                    exit()
        cv2.rectangle(img, start_point, end_point, (0,0,255), 4)
        cv2.imwrite("curr.png",img)
        cv2.imwrite("last.png",last_img)
        fig,ax = plt.subplots(1,3)
        bar = ax[0].imshow(l1)
        ax[0].set_xticks([0,50,100])
        ax[0].set_yticks([0,50,100])
        ax[0].set_xticklabels([-50,0,50])
        ax[0].set_yticklabels([-50,0,50])
        # bar = ax[0].imshow(l1,extent=[-50+320, 50+320, -50+240, 50+240])
        # ax[0].set_xticks([0,320,640])
        # ax[0].set_yticks([0,240,480])
        # ax[0].set_xticklabels([-320,0,320])
        # ax[0].set_yticklabels([-240,0,240])
        ax[0].grid()
        ax[1].imshow(cv2.cvtColor(last_img, cv2.COLOR_BGR2RGB))
        ax[1].set_title("last frame")
        ax[2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[2].set_title("current frame")
        
        # plt.colorbar(bar)
        plt.show()
        start_tracking = False
    
    last_rect_px = img[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    last_img = img
    
    cv2.rectangle(img, start_point, end_point, (0,0,255), 4)
  
    # # calculate the fps
    # frame_count += 1
    # curr_frame_time = time.time()
    # diff = curr_frame_time - last_frame_time
    # if diff > 1:
    #     fps = str(round(frame_count/(diff),2))
    #     last_frame_time = curr_frame_time
    #     frame_count = 0

    # # img, text, location of BLC, font, size, color, thickness, linetype
    # cv2.putText(img, fps+", "+str(int(FRAME_W))+"x"+str(int(FRAME_H)), (7, 30), font, 1, (100, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('window',img)

    # get key
    k = cv2.waitKey(1)

    if k == ord('q'):
        cv2.destroyAllWindows()
        break
    elif k == ord('s'):
        start_tracking = True
