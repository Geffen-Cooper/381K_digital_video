import cv2
import time
import numpy as np
import os

files = os.listdir("palm_imgs")[:-2]

for f in files:
    img = cv2.imread(os.path.join("palm_imgs",f))
    params = f.split("_")

    x,y,w,h = int(params[0]),int(params[1]),int(params[2]),int(params[3].split("-")[0])
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 4)
    while True:
        cv2.imshow("window",img)

        k = cv2.waitKey(1)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break
        if k == ord('n'):
            break
    if k == ord('q'):
            cv2.destroyAllWindows()
            break