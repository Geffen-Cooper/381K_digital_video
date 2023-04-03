import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

frame = cv2.imread("frame.png")
cap = cv2.VideoCapture("Bosphorus_1920x1080_30fps_420_8bit_AVC_MP4.mp4")
ret, frame = cap.read()

def normalize(frame,N,blur=None):
	# get luminance
	frame = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV)
	if blur != None:
		luminance = cv2.GaussianBlur(frame[:,:,0],(7,7),sigmaX=10,sigmaY=10).astype(float)/255
	else:
		luminance = frame[:,:,0].astype(float)/255

	# get the NxN blocks
	h,w = frame.shape[0],frame.shape[1]

	normalized = luminance.copy()
	for row_i,row in enumerate(tqdm(range(h))):
		for col_i,col in enumerate(range(w)):
			block = np.zeros((N,N))
			# get the blocks and normalize the center pixel
			patch_tlx = col - N//2 if col - N//2 > 0 else 0
			patch_tly = row - N//2 if row - N//2 > 0 else 0
			patch_brx = col + N//2 if col + N//2 < w else w
			patch_bry = row + N//2 if row + N//2 < h else h

			block_tlx = N - (patch_brx-patch_tlx)
			block_tly = N - (patch_bry-patch_tly)
			block_brx = N
			block_bry = N

			# print(block_tlx,block_tly,block_brx,block_bry)
			block[block_tly:block_bry,block_tlx:block_brx] = luminance[patch_tly:patch_bry,patch_tlx:patch_brx]

			normalized[row,col] = (luminance[row,col]-block.mean())/(block.std()+1e-9)
	return luminance,normalized

def histogram(img,num_bins):
	min = -3#np.min(img)
	max = 3#np.max(img)
	bin_marks = np.linspace(min,max,num_bins+1)
	bin_lowers = bin_marks[:-1]
	bin_uppers = bin_marks[1:]
	bin_centers = (bin_lowers+bin_uppers)/2
	counts = np.zeros(num_bins)

	i = 0
	for bin_l,bin_u in zip(bin_lowers,bin_uppers):
		in_bin = (img > bin_l) * (img <= bin_u)
		counts[i] = np.sum(in_bin)
		i += 1
	
	counts /= sum(counts)
	return bin_marks,bin_centers,counts


luminance,normalized = normalize(frame,8,blur=True)
cv2.imwrite("luminance.png",luminance*255)
cv2.imwrite("normalized.png",normalized*255)
edges,centers,counts = histogram(normalized,51)
center_l = [round(cent,2) for cent in centers]
plt.bar(centers,counts,width=0.1)
plt.xticks(centers[::4],center_l[::4])

plt.show()
# exit()
while True:
	cv2.imshow("window",luminance)
	cv2.imshow("norm",normalized)
	# quit when click 'q' on keyboard
	# get key
	k = cv2.waitKey(1)

	if k == ord('q'):
		cv2.destroyAllWindows()
		break