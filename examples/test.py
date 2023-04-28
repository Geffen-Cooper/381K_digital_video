import torch
from torchvision import transforms
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import numpy as np
import torch.nn.functional as F

import time
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)


descriptions = ["Swipe Hand Left","Swipe Hand Right","Swipe Hand Up","Swipe Hand Down",\
                "Swipe Two Fingers Left","Swipe Two Fingers Right","Swipe Two Fingers Up","Swipe Two FIngers Down",\
                "Swipe Index Finger Down","Beckon With Hand","Expand Hand","Jazz Hand","One Finger Up","Two Fingers Up","THree Fingers Up",\
                "Lift Hand Up","Move Hand Down","Move Hand Forward","Beckon With Arm","TwoFingers Clockwise","Two Fingers CounterClockwise",
                "Two Fingers Forward","Close Hand","Thumbs Up","OK"]

# Create RNN Model with attention
class AttentionRNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(AttentionRNNModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.device = device

        # RNN
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # attention
        self.attention = torch.nn.Linear(hidden_dim, 1)

        # Readout layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)

        # (batch_size,sequence length) since we want a weight for each hidden state in the sequence
        attention_weights = torch.zeros((x.shape[0],x.shape[1])).to(self.device)

        # output shape is (batch size, sequence length, feature dim)
        output, (hn, cn) = self.rnn(x, (h0, c0))
        
        # for each time step, get the attention weight (do this over the batch)
        for i in range(x.shape[1]):
            attention_weights[:,i] = self.attention(output[:,i,:]).view(-1)
        attention_weights = F.softmax(attention_weights,dim=1)

        # apply attention weights for each element in batch
        attended = torch.zeros(output.shape[0],output.shape[2]).to(self.device)
        for i in range(x.shape[0]):
            attended[i,:] = attention_weights[i]@output[i,:,:]

        return self.fc(attended)
    
# load the model
model = AttentionRNNModel(63,256,1,25,'cpu')
model.load_state_dict(torch.load("model.pth",map_location=torch.device('cpu'))['model_state_dict'])
model.eval()

# initialize the capture object
cap = cv2.VideoCapture(0)
W = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 

# create a window
cv2.namedWindow(str(int(W))+"x"+str(int(H)), cv2.WINDOW_NORMAL)

# measure FPS
last_frame_time = 0
curr_frame_time = 0
frame_count = 0
fps = 0
font = cv2.FONT_HERSHEY_SIMPLEX

count = 0
no_hand_count = 0

hand_in_frame = False

# start loop
i = 0
while True:
    # try to read a frame
    ret,img = cap.read()
    if not ret:
        raise RuntimeError("failed to read frame")

    # flip horizontally
    # image = cv2.flip(image,1)
    image = cv2.resize(img,(320, 240))

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        if hand_in_frame == False:
            hand_in_frame = True
            print("STARTING===========")
            lms_x = ["lmx"+str(i) for i in range(21)]
            lms_y = ["lmy"+str(i) for i in range(21)]
            lms_z = ["lmz"+str(i) for i in range(21)]
            col_names = lms_x + lms_y + lms_z
            df = pd.DataFrame(columns=col_names)
        
        for hand_landmarks in results.multi_hand_landmarks:
            # mp_drawing.draw_landmarks(
            #     img,
            #     hand_landmarks,
            #     mp_hands.HAND_CONNECTIONS,
            #     mp_drawing_styles.get_default_hand_landmarks_style(),
            #     mp_drawing_styles.get_default_hand_connections_style())
            lm_list_x = []
            lm_list_y = []
            lm_list_z = []
            for lm in hand_landmarks.landmark:
                lm_list_x.append(lm.x)
                lm_list_y.append(lm.y)
                lm_list_z.append(lm.z)
        df.loc[len(df.index)] = lm_list_x+lm_list_y+lm_list_z
    elif hand_in_frame == True:
        # print(f"count:{count} --> NOTHING")
        df.loc[len(df.index)] = [0 for j in range(63)]
        no_hand_count += 1

    if no_hand_count == 20 or count == 80:
        hand_in_frame = False

        if count < 30:
            print("False Positive")
            count = 0
            no_hand_count = 0
            continue

        landmarks_seq = df.values
        for col in range(landmarks_seq.shape[1]):
            x = landmarks_seq[:,col]
            # x = medfilt(x,5)
            x = (x-np.min(x))/(np.max(x)-(np.min(x))+1e-6)
            x = (x-np.mean(x))/(np.std(x)+1e-6)
            landmarks_seq[:,col] = x


        landmarks_seq = transforms.ToTensor()(landmarks_seq).float()

        # with torch.no_grad():
        #     model.eval()
        #     pred = F.softmax(model(landmarks_seq))

        # print("prediction:")
        # val,idxs = torch.topk(pred,5)
        # for i in range(5):
        #     print(f"{descriptions[idxs[0][i]]}, {val[0][i]}")
        count = 0
        no_hand_count = 0
    
    if hand_in_frame == True:
        count += 1

    # calculate the fps
    frame_count += 1
    curr_frame_time = time.time()
    diff = curr_frame_time - last_frame_time
    if diff > 1:
        fps = "FPS: " + str(round(frame_count/(diff),2))
        last_frame_time = curr_frame_time
        frame_count = 0
    # img, text, location of BLC, font, size, color, thickness, linetype
    cv2.putText(img, fps, (7, 30), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow(str(int(W))+"x"+str(int(H)),img)
    cv2.imwrite("frame_"+str(i)+".png",img)
    i += 1

    # quit when click 'q' on keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break