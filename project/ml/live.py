import cv2
import time

from torchvision import models,transforms
from tqdm import tqdm
from datasets import *
from models import *

# initialize the capture object
cap = cv2.VideoCapture(0)
W = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 

# create a window
cv2.namedWindow("window", cv2.WINDOW_NORMAL)

# measure FPS
last_frame_time = 0
curr_frame_time = 0
frame_count = 0
fps = 0
font = cv2.FONT_HERSHEY_SIMPLEX


model = models.shufflenet_v2_x0_5(weights='DEFAULT')
model.fc = torch.nn.Linear(1024,4)
model.load_state_dict(torch.load("models/baseline.pth")['model_state_dict'])
model.eval()

center_tlc_x = int(640/2 - 224/2)
center_tlc_y = int(480/2 - 224/2)
center_brc_x = int(640/2 + 224/2)
center_brc_y = int(480/2 + 224/2)
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

    with torch.no_grad():
        input = transforms.ToTensor()(img[start_point[1]:end_point[1],start_point[0]:end_point[0]]).unsqueeze(0)
        out = model(input)*224
        x,y,w,h = int(out[0][0]),int(out[0][1]),int(out[0][2]),int(out[0][3])
    
    print(f"x:{x},y:{y} --> {x+start_point[0]}, {y+start_point[1]}")
    cv2.rectangle(img, (x+start_point[0],y+start_point[1]), (x+w+start_point[0],y+h+start_point[1]), (0,0,255), 4)
    cv2.rectangle(img, start_point, end_point, (0,0,255), 4)
  
    # calculate the fps
    frame_count += 1
    curr_frame_time = time.time()
    diff = curr_frame_time - last_frame_time
    if diff > 1:
        fps = "FPS: " + str(round(frame_count/(diff),2))
        last_frame_time = curr_frame_time
        frame_count = 0
    # img, text, location of BLC, font, size, color, thickness, linetype
    cv2.putText(img, fps+", "+str(int(W))+"x"+str(int(H)), (7, 30), font, 1, (100, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('window',img[start_point[1]:end_point[1],start_point[0]:end_point[0]])

    # quit when click 'q' on keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break