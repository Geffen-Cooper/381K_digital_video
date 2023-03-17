import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

DEBUG_MODE = False

# init rect
RECT_W = 100
RECT_H = 100

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

SEARCH_SIZE = 50
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
shifts = [None,l_shift,tl_shift,t_shift,tr_shift,r_shift,br_shift,b_shift,bl_shift]
news = [None,l_new,tl_new,t_new,tr_new,r_new,br_new,b_new,bl_new]
# start loop
while True:
    # try to read a frame
    ret,img = cap.read()
    if not ret:
        raise RuntimeError("failed to read frame")

    # flip horizontally
    img = cv2.flip(img,1)
    curr_rect_px = img[start_point[1]:end_point[1],start_point[0]:end_point[0]]
    
    eff = False

    # === Rectangle tracking update ===
    # update position
    first_frame = True
    if start_tracking:
        if first_frame == True:
            first_frame = False
            first_frame = last_rect_px
        # l1 = np.ones(9)*1e9
        l1 = np.ones(9)*1e9
        count = 0
        iteration = 0
        x_off = 0
        y_off = 0
        search_locs = all_search_locs
        next_dir = -1
        # keep running the search as long as in the search space
        #print("start")
        while y_off >= -SEARCH_SIZE and y_off <= SEARCH_SIZE and x_off >= -SEARCH_SIZE and x_off <= SEARCH_SIZE:
            print(f"------------iteration: {iteration}")
            # print(search_locs)
            # iterate over the latest diamond points (some already computed)
            for i,coord in enumerate(search_locs):
                count += 1
                # make sure still in picture frame
                if start_point[1]+y_off+coord[0] <= 0 or end_point[1]+y_off+coord[0] >= FRAME_H or start_point[0]+x_off+coord[1] <= 0 or end_point[0]+x_off+coord[1] >= FRAME_W:
                    print("Hitting Boundary")
                else:
                    pred_px = img[start_point[1]+y_off+coord[0]:end_point[1]+y_off+coord[0],start_point[0]+x_off+coord[1]:end_point[0]+x_off+coord[1]]
                    if eff and next_dir > 0:
                        l1[news[next_dir][i]] = np.sum(np.abs(pred_px.astype(int)-last_rect_px.astype(int)))
                    else:
                        l1[i] = np.sum(np.abs(pred_px.astype(int)-last_rect_px.astype(int)))
            last_dir = next_dir
            next_dir = np.argmin(l1)
            print(l1)
            print(f"last: {last_dir}, next: {next_dir}")
            #rint(f"x_off: {x_off}, y_off: {y_off}")
            # need to handle how to update the l1 for next dir, also need to break if next dir is center after one more iter
            # update current offset
            x_off += all_search_locs[next_dir][1]
            y_off += all_search_locs[next_dir][0]

            # update next search locations
            if eff and next_dir != 0:
                search_locs = locs[next_dir] # this seems to be the issue

            # termination case, if center is best
            if last_dir == 0:
                break

            # other wise need to fill in precomputed distances
            else:
                # save differences we can reuse
                if eff and next_dir != 0:
                    for i,shift in enumerate(shifts[next_dir]):
                        l1[shift[1]] = l1[shift[0]]
            iteration += 1
        print(f"best match: ({y_off},{x_off}) after {count} differences")      
        # exit()
        # print(count)
        # closest = np.argmin(l1)
        # col = (closest % (SEARCH_SIZE+SEARCH_SIZE+1))
        # row = (closest // (SEARCH_SIZE+SEARCH_SIZE+1))
        # y_rel = row-SEARCH_SIZE
        # x_rel = col-SEARCH_SIZE
        # print(row,col)
        # print(dirs[closest])
        start_point[0] += x_off
        start_point[1] += y_off
        end_point[0] += x_off
        end_point[1] += y_off
    else:
        min_noise = np.sum(np.abs(curr_rect_px-last_rect_px))

    
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
    # cv2.imwrite("last.png",last_rect_px)
    
    overlay = img.copy()
    cv2.rectangle(overlay, start_point-SEARCH_SIZE, end_point+SEARCH_SIZE, (0,0,255), -1)
    img = cv2.addWeighted(overlay, 0.2, img, 1 - 0.2, 0)
    cv2.putText(img,str(x_off)+","+str(y_off),start_point-10,font, 1, (100, 255, 0), 1, cv2.LINE_AA)
    cv2.rectangle(img, (start_point[0],start_point[1]), end_point, (0,0,255), 4)
    cv2.arrowedLine(img, (start_point[0]-x_off,start_point[1]-y_off), start_point,(100, 255, 0), 3)
  
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
