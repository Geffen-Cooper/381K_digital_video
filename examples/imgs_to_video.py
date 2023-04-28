import os
import cv2
import re

path = "frames"
files = os.listdir(path)
files.sort(key=lambda f: int(re.sub('\D', '', f)))
video_name = 'video2.avi'

frame = cv2.imread(os.path.join(path, files[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for file in files:
    video.write(cv2.imread(os.path.join(path, file)))

cv2.destroyAllWindows()
video.release()