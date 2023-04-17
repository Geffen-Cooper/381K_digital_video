import cv2
import time
import numpy as np

# initialize the capture object
cap = cv2.VideoCapture(0)
W = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 

# create a window
cv2.namedWindow("data collection", cv2.WINDOW_NORMAL)

# measure FPS
last_frame_time = 0
curr_frame_time = 0
frame_count = 0
fps = 0
font = cv2.FONT_HERSHEY_SIMPLEX

# coordinates of center rectangle

# rectangle initial width and height
R_W = 100
R_H = 100

# min and max widths and heights
MIN_R_W = 56
MIN_R_H = 56
MAX_R_W = 224
MAX_R_H = 224

# coordinates of the rectangle
CENTER_TLX = W//2 - R_W//2
CENTER_TLY = H//2 - R_H//2
CENTER_BRX = W//2 + R_W//2
CENTER_BRY = H//2 + R_H//2

# coordinates of the rectangle
start_point = np.array([int(CENTER_TLX),int(CENTER_TLY)])
end_point = np.array([int(CENTER_BRX),int(CENTER_BRY)])

# coordinates of the search space
outer_start_point = start_point-np.abs(np.array([R_W,R_H])-MAX_R_W)//2
outer_end_point = end_point+np.abs(np.array([R_W,R_H])-MAX_R_H)//2

# generate a random rectangle
# f_W: frame width
# f_H: frame height
# min_W: minimum rectangle width
# min_H: minimumm rectangle height
# max_W: minimum rectangle width
# max_H: minimumm rectangle height
# min_WH: minimum width height ratio
# max WH: maximum width height ratio
def get_rand_rect(f_W,f_H,min_W,min_H,max_W,max_H,min_WH,max_WH):
    # get random parameters
    rand_params = np.random.rand(4)
    x = int(rand_params[0]*f_W)
    y = int(rand_params[1]*f_H)
    w = int(rand_params[2]*(max_W-min_W)+min_W)
    wh = rand_params[3]*(max_WH-min_WH)+min_WH
    h = int(w/wh)

    # adjust the values if needed
    if h > max_H:
        h = max_H
        w = h*wh
    elif h < min_H:
        h = min_H
        w = h*wh

    x = x if x + w < f_W else f_W - x
    y = y if y + h < f_H else f_H - y

    return (x,y,w,h)


# start loop
while True:
    # try to read a frame
    ret,img = cap.read()
    if not ret:
        raise RuntimeError("failed to read frame")

    # flip horizontally
    img = cv2.flip(img,1)
    overlay = img.copy()
    img_disp = img.copy()
    cv2.rectangle(overlay, outer_start_point, outer_end_point, (0,0,255), -1)
    img_disp = cv2.addWeighted(overlay, 0.2, img_disp, 1 - 0.2, 0)
    cv2.rectangle(img_disp, start_point, end_point, (0,0,255), 4)
    cv2.circle(img_disp, (start_point[0]+R_W//2,start_point[1]+R_H//2), radius=6, color=(0, 0, 255), thickness=-1)
        
  
    # calculate the fps
    frame_count += 1
    curr_frame_time = time.time()
    diff = curr_frame_time - last_frame_time
    if diff > 1:
        fps = "FPS: " + str(round(frame_count/(diff),2))
        last_frame_time = curr_frame_time
        frame_count = 0
    # img, text, location of BLC, font, size, color, thickness, linetype
    cv2.putText(img_disp, fps+", "+str(int(W))+"x"+str(int(H)), (7, 30), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('data collection',img_disp)

    # quit when click 'q' on keyboard
    # get key
    k = cv2.waitKey(1)

    if k == ord('q'):
        cv2.destroyAllWindows()
        break
    elif k == ord('r'):
        # first save the rectangle in search space coordinate
        cv2.imwrite("palm_imgs\\data\\"+str(start_point[0]-outer_start_point[0])+"_"+str(start_point[1]-outer_start_point[1])+"_"+\
                    str(R_W)+"_"+str(R_H)+"-"+str(time.time())+".png",\
                    img[outer_start_point[1]:outer_end_point[1],outer_start_point[0]:outer_end_point[0]])

        # new random rectangle (in frame coordinates)
        x,y,w,h = get_rand_rect(W,H,65,65,150,150,0.75,1/.75)
        # print(x,y,w,h)

        # rectangle coordinates and w and height
        start_point[0],start_point[1] = x,y
        end_point[0],end_point[1] = x+w,y+h
        R_W,R_H = int(w),int(h)

        # generate random search space coordinates
        outer_start_point[0] = start_point[0]-int(np.random.rand(1)*(MAX_R_W-R_W))
        outer_start_point[1] = start_point[1]-int(np.random.rand(1)*(MAX_R_H-R_H))
        outer_end_point[0] = outer_start_point[0]+MAX_R_W
        outer_end_point[1] = outer_start_point[1]+MAX_R_H

        # adjust if out of the frame
        if outer_start_point[0] < 0:
            outer_start_point[0] = 0
            outer_end_point[0] = outer_start_point[0] + MAX_R_W
        if outer_start_point[1] < 0:
            outer_start_point[1] = 0
            outer_end_point[1] = outer_start_point[1] + MAX_R_H
        if outer_end_point[0] > W:
            outer_end_point[0] = W-1
            outer_start_point[0] = outer_end_point[0] - MAX_R_W
        if outer_end_point[1] > H:
            outer_end_point[1] = H-1
            outer_start_point[1] = outer_end_point[1] - MAX_R_H
        