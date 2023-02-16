import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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
        # print(l1.shape)
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
        overlay = img.copy()
        cv2.rectangle(overlay, start_point-50, end_point+50, (0,0,255), -1)
        img = cv2.addWeighted(overlay, 0.2, img, 1 - 0.2, 0)
        cv2.rectangle(img, start_point, end_point, (0,0,255), 4)
        cv2.circle(img, (start_point[0]+50,start_point[1]+50), radius=6, color=(0, 0, 255), thickness=-1)
        
        cv2.imwrite("curr.png",img)
        cv2.imwrite("last.png",last_img)
        
        closest = np.argmin(l1)
        col = (closest % 101)
        row = (closest // 101)
        y_rel = row-50
        x_rel = col-50
        # print(row,col,x_rel,y_rel,closest)
        cv2.rectangle(img, (start_point[0]+x_rel,start_point[1]+y_rel), (end_point[0]+x_rel,end_point[1]+y_rel), (0,255,0), 4)
        cv2.circle(img, (start_point[0]+50+x_rel,start_point[1]+50+y_rel), radius=6, color=(0, 255, 0), thickness=-1)
        fig = plt.figure(figsize=(10,4))
        ax3d = fig.add_subplot(1, 3, 1, projection='3d')
        ax3d.scatter(x_rel,-y_rel,np.min(l1),c='r')
        ax3d.set_title("relative delta: ("+str(x_rel)+","+str(y_rel)+")")
        ax3d.set_yticks([-50,-25,0,25,50])
        ax3d.set_yticklabels(['50','25','0','-25','-50'])
        ax_before = fig.add_subplot(1,3,2)
        ax_after = fig.add_subplot(1,3,3)
        ax3d.plot_surface(sx,-sy,l1, cmap=cm.coolwarm)
        ax3d.set_xlabel("x - horizontal")
        ax3d.set_ylabel("y - vertical")
        ax_before.imshow(cv2.cvtColor(last_img, cv2.COLOR_BGR2RGB))
        ax_before.set_title("last frame")
        ax_after.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax_after.set_title("current frame")
        ax_after.text(FRAME_W//2-100-40,FRAME_H//2-100,"(-50,-50)",c='w',size=5)
        ax_after.text(FRAME_W//2+100-40,FRAME_H//2-100,"(50,-50)",c='w',size=5)
        ax_after.text(FRAME_W//2-100-40,FRAME_H//2+100,"(-50,50)",c='w',size=5)
        ax_after.text(FRAME_W//2+100-40,FRAME_H//2+100,"(50,50)",c='w',size=5)
        offset = 0
        if x_rel < 0:
            offset += 7
        if y_rel < 0:
            offset += 7
        ax_after.text(FRAME_W//2-25-offset+x_rel,FRAME_H//2-15+y_rel,"("+str(x_rel)+","+str(y_rel)+")",c=(0,1,0),size=5)
        
        # plt.colorbar(bar)
        start_tracking = False
        fig.show()
    
    last_rect_px = img[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    last_img = img
    
    cv2.rectangle(img, start_point, end_point, (0,0,255), 4)
    cv2.circle(img, (start_point[0]+50,start_point[1]+50), radius=6, color=(0, 0, 255), thickness=-1)
  
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
