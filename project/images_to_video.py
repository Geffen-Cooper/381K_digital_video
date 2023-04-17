import os
import cv2
import re

files = os.listdir("gif_frames")
files.sort(key=lambda f: int(re.sub('\D', '', f)))
video_name = 'video.avi'

frame = cv2.imread(os.path.join("gif_frames", files[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 10, (width,height))

for file in files:
    video.write(cv2.imread(os.path.join("gif_frames", file)))

cv2.destroyAllWindows()
video.release()