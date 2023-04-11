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
SEARCH_SIZE = 150

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
sx, sy = np.meshgrid(np.arange(-SEARCH_SIZE,SEARCH_SIZE+1),np.arange(-SEARCH_SIZE,SEARCH_SIZE+1))

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



def diamond_search(last_frame, reference_patch):
    last_rect_px = reference_patch
    img = last_frame
    sx, sy = np.meshgrid(np.arange(-SEARCH_SIZE,SEARCH_SIZE+1),np.arange(-SEARCH_SIZE,SEARCH_SIZE+1))

    x_off = 0
    y_off = 0

    all_search_locs = np.array([[0,0],[0,-2],[-1,-1],[-2,0],[-1,1],[0,2],[1,1],[2,0],[1,-1]])
    c_search_locs = np.array([[0,-1],[-1,0],[0,1],[1,0]])
    l_search_locs = np.array([[2,0],[1,-1],[0,-2],[-1,-1],[-2,0]])
    tl_search_locs = np.array([[0,-2],[-1,-1],[-2,0]])
    t_search_locs = np.array([[0,-2],[-1,-1],[-2,0],[-1,1],[0,2]])
    tr_search_locs = np.array([[-2,0],[-1,1],[0,2]])
    r_search_locs = np.array([[-2,0],[-1,1],[0,2],[1,1],[2,0]])
    br_search_locs = np.array([[0,2],[1,1],[2,0]])
    b_search_locs = np.array([[0,2],[1,1],[2,0],[1,-1],[0,-2]])
    bl_search_locs = np.array([[2,0],[1,-1],[0,-2]])
    locs = [c_search_locs,l_search_locs,tl_search_locs,t_search_locs,tr_search_locs,r_search_locs,br_search_locs,b_search_locs,bl_search_locs]

    c_shift = None
    c_new = np.array([1,2,3,4])
    l_shift = np.array([[0,5],[1,0],[2,4],[8,6]])
    l_new = np.array([7,8,1,2,3])
    tl_shift = np.array([[4,5],[3,4],[0,6],[2,0],[8,7],[1,8]])
    tl_new = np.array([1,2,3])
    t_shift = np.array([[0,7],[3,0],[4,6],[2,8]])
    t_new = np.array([1,2,3,4,5])
    tr_shift = np.array([[0,8],[4,0],[2,1],[3,2],[6,7],[5,6]])
    tr_new = np.array([3,4,5])
    r_shift = np.array([[0,1],[4,2],[5,0],[6,8]])
    r_new = np.array([3,4,5,6,7])
    br_shift = np.array([[0,2],[6,0],[8,1],[4,3],[5,4],[7,8]])
    br_new = np.array([5,6,7])
    b_shift = np.array([[0,3],[8,2],[7,0],[6,4]])
    b_new = np.array([5,6,7,8,1])
    bl_shift = np.array([[0,4],[8,0],[2,3],[1,2],[6,5],[7,6]])
    bl_new = np.array([7,8,1])
    shifts = [c_shift,l_shift,tl_shift,t_shift,tr_shift,r_shift,br_shift,b_shift,bl_shift]
    news = [c_new,l_new,tl_new,t_new,tr_new,r_new,br_new,b_new,bl_new]

    rel_locs = []

    l1 = np.ones(9)*1e9
    count = 0
    iteration = 0
    x_off = 0
    y_off = 0
    search_locs = all_search_locs
    next_dir = -1

    # keep running the search as long as in the search space
    while y_off >= -SEARCH_SIZE and y_off <= SEARCH_SIZE and x_off >= -SEARCH_SIZE and x_off <= SEARCH_SIZE:
        
        # iterate over the latest diamond points (some already computed)
        for i,coord in enumerate(search_locs):
            count += 1
            # make sure still in picture frame
            if start_point[1]+y_off+coord[0] <= 0 or end_point[1]+y_off+coord[0] >= FRAME_H or start_point[0]+x_off+coord[1] <= 0 or end_point[0]+x_off+coord[1] >= FRAME_W:
                print("Hitting Boundary")
            else:
                pred_px = img[start_point[1]+y_off+coord[0]:end_point[1]+y_off+coord[0],start_point[0]+x_off+coord[1]:end_point[0]+x_off+coord[1]]
                if next_dir >= 0:
                    l1[news[next_dir][i]] = np.mean(np.abs(pred_px.astype(int)-last_rect_px.astype(int)))
                    if next_dir == 0:
                        l1[5:] = 1e9
                else:
                    l1[i] = np.mean(np.abs(pred_px.astype(int)-last_rect_px.astype(int)))
        last_dir = next_dir
        next_dir = np.argmin(l1)

        
        # update current offset
        if last_dir == 0: # we handle the center case differently since the inner diamond is different
            if next_dir != 0: # only add an offset if not in the center
                print(next_dir)
                x_off += c_search_locs[next_dir-1][1]
                y_off += c_search_locs[next_dir-1][0] 
        else:
            x_off += all_search_locs[next_dir][1]
            y_off += all_search_locs[next_dir][0]

        # update next search locations
        search_locs = locs[next_dir]

        rel_locs.append([y_off,x_off,np.min(l1)])

        # termination case, if center is best
        if last_dir == 0:
            break

        # other wise need to fill in precomputed distances
        else:
            # save differences we can reuse
            if next_dir != 0:
                for i,shift in enumerate(shifts[next_dir]):
                    l1[shift[1]] = l1[shift[0]]
        iteration += 1
    
    return rel_locs


# start loop
while True:
    # try to read a frame
    ret,img = cap.read()
    if not ret:
        raise RuntimeError("failed to read frame")

    # flip horizontally
    img = cv2.flip(img,1)
    # imgb = cv2.GaussianBlur(img,(7,7),2,2)

    # === Rectangle tracking update ===
    # update position
    if start_tracking:
        l1 = np.zeros((sy.shape[0],sx.shape[1]))
        # l1b = np.zeros((sy.shape[0],sx.shape[1]))
        # print(l1.shape)
        # get the rows of sx and sy
        for i,(xr,yr) in enumerate(zip(sx,sy)):
            # get the columns of xr and yr
            for j,(xc,yc) in enumerate(zip(xr,yr)):
                try:
                    pred_px = img[start_point[1]+yc:end_point[1]+yc,start_point[0]+xc:end_point[0]+xc]
                    # pred_pxb = imgb[start_point[1]+yc:end_point[1]+yc,start_point[0]+xc:end_point[0]+xc]
                    l1[i,j] = np.mean(np.abs(pred_px.astype(int)-last_rect_px.astype(int)))
                    # l1b[i,j] = np.sum(np.abs(pred_pxb.astype(int)-last_rect_px.astype(int)))
                except:
                    print(start_point,end_point)
                    print(i,j,xc,yc)
                    print(pred_px.shape)
                    print(last_rect_px.shape)
                    exit()
        diamond_points = np.array(diamond_search(img,last_rect_px))
        print(diamond_points[-1])
        overlay = img.copy()
        cv2.rectangle(overlay, start_point-SEARCH_SIZE, end_point+SEARCH_SIZE, (0,0,255), -1)
        img = cv2.addWeighted(overlay, 0.2, img, 1 - 0.2, 0)
        cv2.rectangle(img, start_point, end_point, (0,0,255), 4)
        cv2.circle(img, (start_point[0]+50,start_point[1]+50), radius=6, color=(0, 0, 255), thickness=-1)
        
        cv2.imwrite("curr.png",img)
        cv2.imwrite("last.png",last_img)
        
        closest = np.argmin(l1)
        col = (closest % (SEARCH_SIZE+SEARCH_SIZE+1))
        row = (closest // (SEARCH_SIZE+SEARCH_SIZE+1))
        y_rel = row-SEARCH_SIZE
        x_rel = col-SEARCH_SIZE
        # print(row,col,x_rel,y_rel,closest)
        cv2.rectangle(img, (start_point[0]+x_rel,start_point[1]+y_rel), (end_point[0]+x_rel,end_point[1]+y_rel), (0,255,0), 4)
        cv2.circle(img, (start_point[0]+50+x_rel,start_point[1]+50+y_rel), radius=6, color=(0, 255, 0), thickness=-1)
        fig = plt.figure(figsize=(14,4))
        ax3d = fig.add_subplot(1, 3, 1, projection='3d')
        # ax3db = fig.add_subplot(1, 4, 2, projection='3d')
        ax3d.scatter(x_rel,-y_rel,np.min(l1),c='r')
        ax3d.scatter(diamond_points[:,1],-diamond_points[:,0],diamond_points[:,2]+1,c='k',s=2)
        ax3d.set_title("relative delta: ("+str(x_rel)+","+str(y_rel)+")")
        ax3d.set_yticks([-SEARCH_SIZE,-SEARCH_SIZE/2,0,SEARCH_SIZE/2,SEARCH_SIZE])
        ax3d.set_yticklabels([str(SEARCH_SIZE),str(SEARCH_SIZE/2),'0',str(-SEARCH_SIZE/2),str(-SEARCH_SIZE)])
        ax3d.set_xticks([-SEARCH_SIZE,-SEARCH_SIZE/2,0,SEARCH_SIZE/2,SEARCH_SIZE])
        ax3d.set_xticklabels([str(-SEARCH_SIZE),str(-SEARCH_SIZE/2),'0',str(SEARCH_SIZE/2),str(SEARCH_SIZE)])
        # ax3db.set_zlim([0,1.5e6])
        # ax3d.set_zlim([0,1.5e6])
        ax_before = fig.add_subplot(1,3,2)
        ax_after = fig.add_subplot(1,3,3)
        ax3d.plot_surface(sx,-sy,l1, cmap=cm.coolwarm,alpha=0.35,edgecolor='gray', lw=0.5, rstride=8, cstride=8,)
        ax3d.contour(sx, -sy, l1, zdir='z', offset=10, cmap='coolwarm')
        # ax3db.plot_surface(sx,-sy,l1b, cmap=cm.coolwarm)
        ax3d.set_xlabel("x - horizontal")
        ax3d.set_ylabel("y - vertical")
        ax_before.imshow(cv2.cvtColor(last_img, cv2.COLOR_BGR2RGB))
        ax_before.set_title("last frame")
        ax_after.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax_after.set_title("current frame")
        ax_after.text(FRAME_W//2-(SEARCH_SIZE+RECT_W//2)-40,FRAME_H//2-(SEARCH_SIZE+RECT_H//2),"("+str(-SEARCH_SIZE)+","+str(-SEARCH_SIZE)+")",c='w',size=5)
        ax_after.text(FRAME_W//2+(SEARCH_SIZE+RECT_W//2)-40,FRAME_H//2-(SEARCH_SIZE+RECT_H//2),"("+str(SEARCH_SIZE)+","+str(-SEARCH_SIZE)+")",c='w',size=5)
        ax_after.text(FRAME_W//2-(SEARCH_SIZE+RECT_W//2)-40,FRAME_H//2+(SEARCH_SIZE+RECT_H//2),"("+str(-SEARCH_SIZE)+","+str(SEARCH_SIZE)+")",c='w',size=5)
        ax_after.text(FRAME_W//2+(SEARCH_SIZE+RECT_W//2)-40,FRAME_H//2+(SEARCH_SIZE+RECT_H//2),"("+str(SEARCH_SIZE)+","+str(SEARCH_SIZE)+")",c='w',size=5)
        offset = 0
        if x_rel < 0:
            offset += 7
        if y_rel < 0:
            offset += 7
        ax_after.text(FRAME_W//2-25-offset+x_rel,FRAME_H//2-15+y_rel,"("+str(x_rel)+","+str(y_rel)+")",c=(0,1,0),size=5)
        
        # plt.colorbar(bar)
        start_tracking = False
        # print(l1[:5,:5])
        # print(l1b[:5,:5])
        fig.show()
    
    last_imgb = cv2.GaussianBlur(img,(7,7),0.5,0.5)
    last_rect_px = img[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    last_rect_pxb = last_imgb[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    last_img = img
    cv2.imwrite("last_rec.png",last_rect_px)
    
    cv2.rectangle(img, start_point, end_point, (0,0,255), 4)
    cv2.circle(img, (start_point[0]+RECT_W//2,start_point[1]+RECT_H//2), radius=6, color=(0, 0, 255), thickness=-1)
  
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
