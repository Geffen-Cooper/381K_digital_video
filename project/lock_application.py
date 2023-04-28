import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from torchvision import models,transforms
from tqdm import tqdm
import torch

# init rect
RECT_W = 110
RECT_H = 110
SEARCH_SIZE = 112

# initialize the capture object
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # this is the magic!

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
FRAME_W = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
FRAME_H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 

# create a window
cv2.namedWindow("("+str(int(FRAME_W))+"x"+str(int(FRAME_H))+")", cv2.WINDOW_NORMAL)

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

model = models.shufflenet_v2_x0_5(weights='DEFAULT')
model.fc = torch.nn.Linear(1024,4)
model.load_state_dict(torch.load("ml/models/baseline_finetune.pth")['model_state_dict'])
model.eval()

start_tracking = False

got_wrong = False
got_right = False

# try to read a frame
ret,img = cap.read()
if not ret:
    raise RuntimeError("failed to read frame")

# flip horizontally
img = cv2.flip(img,1)
last_rect_px = img[start_point[1]:end_point[1],start_point[0]:end_point[0]]


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

running_l1 = 0
running_count = 0
max_l1 = 0
alpha = 0.75
first_frame = True
reset = False
first_rect = None
box_color = (0,150,0)
arrow_color = (0,0,255)


#lock_input_coords = [[0,0],[0,1],[0,2],[1,2],[2,3]]
lock_input_coords = [[2,2]]
edge_len_y = int(FRAME_H // 6)
edge_locs_y = np.arange(0,FRAME_H)[::edge_len_y]
edge_locs_x_offset = int(FRAME_W//2 - 2*edge_len_y)

cursor_pos = (start_point[0]+RECT_W//2,start_point[1]+RECT_H//2)

lock_points = np.zeros((5,5,2))
for row in range(5):
    for col in range(5):
        lock_points[row,col,:] = [(row+1)*edge_len_y,col*edge_len_y+edge_locs_x_offset]
lock_done = False

lock_code = np.array([[2,2],[1,2],[1,3],[2,3],[3,3],[3,2],[3,1],[2,1],[1,1],[0,1],[0,2],[0,3],[0,4],[1,4],\
                      [2,4],[3,4],[4,4],[4,3],[4,2],[4,1],[4,0],[3,0],[2,0],[1,0],[0,0]])

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
    if start_tracking and lock_done == False:
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
                    # print(next_dir)
                    x_off += c_search_locs[next_dir-1][1]
                    y_off += c_search_locs[next_dir-1][0] 
            else:
                x_off += all_search_locs[next_dir][1]
                y_off += all_search_locs[next_dir][0]

            # update next search locations
            search_locs = locs[next_dir]

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
        
        # store some stats
        if first_frame == True:
            first_frame = False
            running_l1 = np.min(l1)
            running_count = count
            max_l1 = running_l1
            first_rect = last_rect_px
        else:
            running_l1 = running_l1*alpha + np.min(l1)*(1-alpha)
            running_count = running_count*alpha + count*(1-alpha)
            if np.min(l1) > max_l1:
                max_l1 = np.min(l1)
        if np.min(l1) > 6 and np.min(l1) < 16:
            # box_color = (0,0,255)
            center = (start_point[0]+(end_point[0]-start_point[0])//2,start_point[1]+(end_point[1]-start_point[1])//2)
            with torch.no_grad():
                search_tlx = center[0]-SEARCH_SIZE
                search_tly = center[1]-SEARCH_SIZE
                search_brx = center[0]+SEARCH_SIZE
                search_bry = center[1]+SEARCH_SIZE
                # check collision
                if search_tlx <= 0: # left
                    search_tlx = 0
                    search_brx = 2*SEARCH_SIZE
                if search_tly <= 0: # top
                    search_tly = 0
                    search_bry = 2*SEARCH_SIZE
                if search_brx >= FRAME_W: # right
                    search_brx = FRAME_W
                    search_tlx = FRAME_W - 2*SEARCH_SIZE
                if search_bry >= FRAME_H: # bottom
                    search_bry = FRAME_H
                    search_tly = FRAME_H - 2*SEARCH_SIZE
                input = transforms.ToTensor()(img[int(search_tly):int(search_bry),int(search_tlx):int(search_brx)]).unsqueeze(0)
                print(center,search_tlx,search_brx,search_tly,search_bry,input.shape)
                out = model(input)*224
                x,y,w,h = int(out[0][0]),int(out[0][1]),int(out[0][2]),int(out[0][3])
                # first_rect = img[center[1]-SEARCH_SIZE+y:center[1]-SEARCH_SIZE+y+h,\
                #                  center[0]-SEARCH_SIZE+x:center[0]-SEARCH_SIZE+x+w]
                start_point[0] = center[0]-SEARCH_SIZE+x
                start_point[1] = center[1]-SEARCH_SIZE+y
                end_point[0] = center[0]-SEARCH_SIZE+x+w
                end_point[1] = center[1]-SEARCH_SIZE+y+h

                RECT_W = w
                RECT_H = h

                # check collision
                if start_point[0] <= 0: # left
                    start_point[0] = 0
                    end_point[0] = w
                if start_point[1] <= 0: # top
                    start_point[1] = 0
                    end_point[1] = h
                if end_point[0] >= FRAME_W: # right
                    end_point[0] = FRAME_W
                    start_point[0] = FRAME_W - w
                if end_point[1] >= FRAME_H: # bottom
                    end_point[1] = FRAME_H
                    start_point[1] = FRAME_H - h

                first_rect = img[start_point[1]:end_point[1],start_point[0]:end_point[0]]
        elif np.min(l1) > 10:
            box_color = (0,0,255)
        else:
            box_color = (0,150,0)
        # print(f"best match: ({y_off},{x_off}) after {count} diffs, curr diff: {np.min(l1)}, running diff: {int(running_l1)}, max diff: {int(max_l1)}") 
       
        # update the position after tracking
        start_point[0] += x_off
        start_point[1] += y_off
        end_point[0] += x_off
        end_point[1] += y_off

    # after tracking done, display results
    
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
    if not first_frame:
        last_rect_px = first_rect
    # cv2.imwrite("last.png",last_rect_px)
    
    overlay = img.copy()
    img_disp = img.copy()
    center = np.array([start_point[0]+RECT_W//2,start_point[1]+RECT_H//2])
    cv2.rectangle(overlay, center-SEARCH_SIZE, center+SEARCH_SIZE, box_color, -1)
    img_disp = cv2.addWeighted(overlay, 0.1, img_disp, 1 - 0.1, 0)
    cv2.putText(img_disp,str(x_off)+","+str(y_off),start_point-10,font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(img_disp, (start_point[0],start_point[1]), end_point, box_color, 4)
    # cv2.arrowedLine(img_disp, (start_point[0]+RECT_W//2-x_off,start_point[1]+RECT_H//2-y_off), (start_point[0]+RECT_W//2,start_point[1]+RECT_H//2),(0, 0, 0), 3)
    cv2.circle(img_disp,(start_point[0]+RECT_W//2,start_point[1]+RECT_H//2),4,box_color)
  
    # calculate the fps
    frame_count += 1
    curr_frame_time = time.time()
    diff = curr_frame_time - last_frame_time
    if diff > 1:
        fps = str(round(frame_count/(diff),1))
        last_frame_time = curr_frame_time
        frame_count = 0

    # img, text, location of BLC, font, size, color, thickness, linetype
    cv2.putText(img_disp, "FPS: " +fps, (7, 30), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    
    cv2.putText(img_disp, "Avg. Error:", (300-40, 23), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(img_disp,(300+50,8),(620,30),(0,0,0),1)
    cv2.rectangle(img_disp,(304+50,12),(304+50+int((615-305-50)*running_l1/100),26),box_color,-1)
    cv2.rectangle(img_disp,(300+50,8),(305+50+int((615-305-50)*20/100),30),(0,0,0),1)

    cv2.putText(img_disp, "Avg. Diffs: "+str(round(running_count,1)), (300-90, 23+32), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(img_disp,(300+50,8+30),(620,30+30),(0,0,0),1)
    cv2.rectangle(img_disp,(304+50,12+30),(304+50+int((615-305)*running_count/500),26+30),(0,0,0),-1)
    
    # if lock_done == True:
    #     img_disp = img.copy()

    # draw lock pattern
    edge_len_y = int(FRAME_H // 6)
    edge_locs_y = np.arange(0,FRAME_H)[::edge_len_y]
    edge_locs_x_offset = int(FRAME_W//2 - 2*edge_len_y)

    cursor_pos = np.array([start_point[0]+RECT_W//2,start_point[1]+RECT_H//2])
    cursor_pos_comp = np.array([cursor_pos[1],cursor_pos[0]])

    for row in range(5):
        for col in range(5):
            cv2.circle(img_disp,(col*edge_len_y+edge_locs_x_offset,(row+1)*edge_len_y),9,(0,0,0),2)
            # check if cursor in this position
            point_pos = np.array([(row+1)*edge_len_y,col*edge_len_y+edge_locs_x_offset])
            dist_to_center = np.sqrt(np.sum(np.square(cursor_pos_comp-point_pos)))
            
            if dist_to_center < 9 and np.array([row,col] != lock_input_coords[-1]) and not got_right:
                lock_input_coords.append([row,col])


    for i,selected in enumerate(lock_input_coords):
        # last coord goes to cursor
        if i + 1 == len(lock_input_coords):
            if not got_right:
                coords_start = lock_points[lock_input_coords[i][0],lock_input_coords[i][1]]
                cv2.arrowedLine(img_disp, (int(coords_start[1]),int(coords_start[0])), cursor_pos,arrow_color, 2)
        else:
            coords_start = lock_points[lock_input_coords[i][0],lock_input_coords[i][1]]
            coords_end = lock_points[lock_input_coords[i+1][0],lock_input_coords[i+1][1]]
            cv2.arrowedLine(img_disp, (int(coords_start[1]),int(coords_start[0])), (int(coords_end[1]),int(coords_end[0])),arrow_color, 2)

    if got_wrong and not start_tracking and not got_right:
        cv2.putText(img_disp, "WRONG, Try Again", (175,455), font, 1, (0, 0, 255), 2, cv2.LINE_AA)  
    if got_right:
        cv2.putText(img_disp, "UNLOCKED", (235,455), font, 1, (0, 150, 0), 2, cv2.LINE_AA)  
        start_tracking = False
    cv2.imshow("("+str(int(FRAME_W))+"x"+str(int(FRAME_H))+")",img_disp)

    if box_color == (0,0,255):
        lock_done = True
        RECT_H = 110
        RECT_W = 110
        # print(np.array(lock_input_coords)==lock_code)
        if len(lock_input_coords) == len(lock_code) and (np.array(lock_input_coords) == lock_code).all():
            got_right = True
            start_point = np.array([center_tlc_x, center_tlc_y])
            end_point = np.array([center_brc_x, center_brc_y])
            box_color = (0,150,0)
            arrow_color = (0,150,0)
        else:
            lock_input_coords = [[2,2]]
            start_tracking = False
            lock_done = False
            box_color = (0,150,0)
            running_l1 = 0
            running_count = 0
            max_l1 = 0
            first_frame = True
            first_rect = None
            x_off = 0
            y_off = 0
            start_point = np.array([center_tlc_x, center_tlc_y])
            end_point = np.array([center_brc_x, center_brc_y])
            got_wrong = True
      

    # get key
    k = cv2.waitKey(1)

    if k == ord('q'):
        cv2.destroyAllWindows()
        break
    elif k == ord('s'):
        start_tracking = True
