import cv2
import numpy as np

FRAME_H = 480
FRAME_W = 640

# blurred = cv2.imread("video_frames/frame_7.png")[100:250,270:420]
# print(blurred.shape)
# cv2.imwrite("blur_h.png",blurred)
# while True:
#     cv2.imshow("window",blurred)
#     # get key
#     k = cv2.waitKey(1)

#     if k == ord('q'):
#         cv2.destroyAllWindows()
#         break

blur_h = cv2.imread("blur_h.png")
img = cv2.imread("video_frames/frame__3.png")
img = cv2.imread("blurred.png")
img_ = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]/255
blur_h = 255-cv2.cvtColor(blur_h, cv2.COLOR_BGR2YUV)[:,:,0]
blur_h = np.zeros((11,11)).astype(float)
blur_h[:,5] = 1/11
img_ = cv2.GaussianBlur(img_,(3,3),3,3)


while True:
    cv2.imshow("window",blur_h)
    cv2.imshow("window2",img_)
    # get key
    k = cv2.waitKey(1)

    if k == ord('q'):
        cv2.destroyAllWindows()
        break

TOTAL_H = int(FRAME_H+blur_h.shape[0]-1)
TOTAL_W = int(FRAME_W+blur_h.shape[1]-1)

filter = np.zeros((int(FRAME_H+blur_h.shape[0]-1),int(FRAME_W+blur_h.shape[1]-1)), float)
input = np.zeros((int(FRAME_H+blur_h.shape[0]-1),int(FRAME_W+blur_h.shape[1]-1)), float)

filter[int(TOTAL_H//2-blur_h.shape[0]//2-1):int(TOTAL_H//2-blur_h.shape[0]//2-1+blur_h.shape[0]),\
       int(TOTAL_W//2-blur_h.shape[0]//2-1):int(TOTAL_W//2-blur_h.shape[0]//2-1+blur_h.shape[0])] = blur_h
input[blur_h.shape[0]//2:int(blur_h.shape[0]//2+FRAME_H),blur_h.shape[0]//2:int(blur_h.shape[0]//2+FRAME_W)] = img_

while True:
    cv2.imshow("window",input)
    cv2.imshow("window2",filter)
    # get key
    k = cv2.waitKey(1)

    if k == ord('q'):
        cv2.destroyAllWindows()
        break

F1 = np.fft.fft2(filter)
F2 = np.fft.fft2(input)

F3 = F2/F1#F1*F2
_img = np.fft.ifft2(F3)
_img = np.real(np.fft.fftshift(_img))

while True:
    cv2.imshow("window",_img)
    # get key
    k = cv2.waitKey(1)

    if k == ord('q'):
        cv2.destroyAllWindows()
        break

cv2.imwrite("blurred.png",_img[blur_h.shape[0]//2:int(blur_h.shape[0]//2+FRAME_H),blur_h.shape[0]//2:int(blur_h.shape[0]//2+FRAME_W)]*255)