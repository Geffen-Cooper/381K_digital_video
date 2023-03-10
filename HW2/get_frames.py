import numpy as np
import cv2
import torch 
from models import*

def get_video_frames(video_path,is_img_frames,patch_size=None,patch_coord=None,num_frames=None):
    if is_img_frames == False:
        # create video capture object
        cap = cv2.VideoCapture(video_path)
            
        # Check if video opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video file")
            exit()

        # read the first frame
        ret, frame = cap.read()
        frame = cv2.resize(frame,(patch_size[0],patch_size[1]))

        if ret:
            # by defaut return the first frame
            if patch_size == None and patch_coord == None and num_frames == None:
                # release the video capture object
                cap.release()
                return frame

            else:
                if num_frames == None:
                    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # create frame buffer
                frames = np.empty((num_frames, patch_size[1], patch_size[0], 3), np.dtype('uint8'))
                frame_count = 0

                # store the first frame
                start_x,start_y = patch_coord[0], patch_coord[1]
                end_x, end_y = start_x + patch_size[0], start_y + patch_size[1]

                frames[frame_count] = frame[start_y:end_y,start_x:end_x,:]
                frame_count += 1

                # store the remaning frames
                while (frame_count < num_frames and ret):
                    start_x,start_y = patch_coord[0], patch_coord[1]
                    end_x, end_y = start_x + patch_size[0], start_y + patch_size[1]
                    try:
                        frames[frame_count] = frame[start_y:end_y,start_x:end_x,:]
                    except:
                        print(frame_count)
                        print(frame,ret)
                    frame_count += 1
                    ret, frame = cap.read()
                    frame = cv2.resize(frame,(patch_size[0],patch_size[1]))
                return frames
        else:
            print("can't get first frame")
            exit()
    else:
        files = video_path
        num_frames = len(files)

        # create frame buffer
        frames = np.empty((num_frames, patch_size[1], patch_size[0], 3), np.dtype('uint8'))
        frame_count = 0

        for file in files:
            frame = cv2.imread(file)
            
            # store the first frame
            start_x,start_y = patch_coord[0], patch_coord[1]
            end_x, end_y = start_x + patch_size[0], start_y + patch_size[1]

            frames[frame_count] = frame[start_y:end_y,start_x:end_x,:]
            frame_count += 1
        return frames

res_x = 1280//2
res_y = 720//2
before = get_video_frames("test.mp4",False,(res_x,res_y),(0,0))
print(before.shape)
after = np.copy(before)

after = torch.tensor(after.transpose(3,0,1,2)).float().div(255.0)
after = after.unsqueeze(0).to('cuda')
after_proc = torch.clone(after)
model = ArtifactReduction()
model.load_state_dict(torch.load("models/Small_LR.pth")['model_state_dict'])
with torch.no_grad():
    if model != None: 
        model.to('cuda')
        frame_depth = 5
        patch_size = (res_x,res_y)
        # go over each frame
        for depth_coord in range(before.shape[0]):
            # if at the beginning add padding
            if depth_coord < frame_depth//2:
                frames_proc = torch.cat([torch.zeros((1,3,frame_depth//2-depth_coord,patch_size[1],patch_size[0])).to('cuda'),after[:,:,:depth_coord+frame_depth//2+1,:,:]],dim=2)
            # if at the end add padding
            elif (before.shape[0]-1-depth_coord) < frame_depth//2:
                frames_proc = torch.cat([after[:,:,depth_coord-frame_depth//2:,:,:],torch.zeros((1,3,depth_coord-(before.shape[0]-1)+frame_depth//2,patch_size[1],patch_size[0])).to('cuda')],dim=2)
            # otherwise get frames [t-2,t-1,t,t+1,t+2] where t is ground truth
            else:
                # get frame_depth frames for training, and the ground truth is the middle frame
                frames_proc = after[:,:,depth_coord-frame_depth//2:depth_coord+frame_depth//2+1,:,:]
        
            # forward, output is a single frame
            output = model(frames_proc)
            after_proc[:,:,depth_coord,:,:] = (output-output.min())/(output.max()-output.min())

aft = (torch.permute(after_proc[0].to('cpu'),(1,2,3,0)).numpy()*255).astype(np.uint8)

# print(comp.shape,aft.shape)
out_comp = cv2.VideoWriter('out_comp.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (res_x,res_y))
out_aft = cv2.VideoWriter('out_aft.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (res_x,res_y))
# out_gt = cv2.VideoWriter('out_gt.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, self.patch_size)

i = 0
for frame_comp, frame_aft in zip(before,aft):
    out_comp.write(frame_comp)
    out_aft.write(frame_aft)

out_comp.release()
out_aft.release()
# out_gt.release()


cap_comp = cv2.VideoCapture("out_comp.avi")
cap_aft = cv2.VideoCapture("out_aft.avi")
# cap_gt = cv2.VideoCapture("out_gt.avi")

while(True):
    ret_comp, frame_comp = cap_comp.read()
    ret_aft, frame_aft = cap_aft.read()
    # ret_gt, frame_gt = cap_gt.read()

    if ret_comp == True and ret_aft == True:#ret_gt == True: 
        
        # Display the resulting frame    
        cv2.imshow('comp',frame_comp)
        cv2.imshow('aft',frame_aft)
        # cv2.imshow('gt',frame_gt)
        
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
    
    # Break the loop
    else:
        cap_comp.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap_aft.set(cv2.CAP_PROP_POS_FRAMES, 0)