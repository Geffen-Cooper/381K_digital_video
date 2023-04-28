import cv2
import time
import numpy as np
import os

files = os.listdir("palm_imgs\\data")
font = cv2.FONT_HERSHEY_SIMPLEX

count = 0
displayed = False
while True:
    if displayed == False:
        img = cv2.imread(os.path.join("palm_imgs\\data",files[count]))
        params = files[count].split("_")

        x,y,w,h = int(params[0]),int(params[1]),int(params[2]),int(params[3].split("-")[0])
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 4)
        cv2.putText(img, "("+str(count+1)+"/"+str(len(files))+")"+f" x:{x}, y:{y}, w:{w}, h:{h}", (0, 20), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow(f"{img.shape}",img)
        displayed = True
        if img.shape != (224,224,3):
            print(files[count])

    k = cv2.waitKey(1)
    if k == ord('q'):
        cv2.destroyAllWindows()
        break
    if k == ord('n'):
        count += 1
        if count == len(files):
            count = 0
        displayed = False
        cv2.imwrite(f"collected_{count}.png",img)
    if k == ord('b'):
        count -= 1
        if count == -1:
            count = len(files)-1
        displayed = False