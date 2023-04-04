import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# this function reads a single frame from a file
def read_frame(video_file,w,h):
	num_bytes = int(w*h)
	yuv_file = open(video_file,'rb')
	frame_shape = (h,w)
	bytes = yuv_file.read(num_bytes)
	frame = np.frombuffer(bytes,np.uint8)
	frame = frame.reshape(frame_shape)
	return frame

def normalize(luminance,N,blur=None):
	if blur != None:
		luminance = cv2.GaussianBlur(luminance,(5,5),sigmaX=blur,sigmaY=blur)

	# get the NxN blocks
	h,w = luminance.shape[0],luminance.shape[1]

	normalized = np.log(luminance.copy().astype(float))
	for row_i,row in enumerate(tqdm(range(h))):
		for col_i,col in enumerate(range(w)):
			block = np.zeros((N,N)).astype(float)
			# get the blocks and normalize the center pixel
			patch_tlx = col - N//2 if col - N//2 > 0 else 0
			patch_tly = row - N//2 if row - N//2 > 0 else 0
			patch_brx = col + N//2 + 1 if col + N//2 + 1 <= w else w
			patch_bry = row + N//2 + 1 if row + N//2 + 1 <= h else h

			block_tlx = 0 if col - N//2 > 0 else N - (patch_brx-patch_tlx)
			block_tly = 0 if row - N//2 > 0 else N - (patch_bry-patch_tly)
			block_brx = N if col + N//2 + 1 <= w else (patch_brx-patch_tlx)
			block_bry = N if row + N//2 + 1 <= h else (patch_bry-patch_tly)

			# print(f"patch -- tlx:{patch_tlx},tly:{patch_tly},brx:{patch_brx},bry:{patch_bry}")
			# print(f"block -- tlx:{block_tlx},tly:{block_tly},brx:{block_brx},bry:{block_bry}")
			block[block_tly:block_bry,block_tlx:block_brx] = luminance[patch_tly:patch_bry,patch_tlx:patch_brx]
			normalized[row,col] = (luminance[row,col]-block.mean())/(block.std()+1e-3)
	return luminance,normalized

def histogram(img,num_bins):
	min = np.min(img)
	max = np.max(img)
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
	
	counts /= np.max(counts)
	return bin_marks,bin_centers,counts


# read frame
sigmas= [None,2,5,10,20]
frame = read_frame("Jockey_1920x1080_120fps_420_8bit_YUV.yuv",1920,1080)

for sigma in sigmas:
	# do divisive normalization
	luminance,normalized = normalize(frame,5,blur=sigma)
	edges,centers,counts = histogram(normalized,201)

	# plot histogram
	# plt.bar(centers,counts,width=0.05)
	# center_l = ["-3","-2","-1","0","1","2","3"]
	# centers = [-3,-2,-1,0,1,2,3]
	# plt.xticks(centers,center_l)
	# plt.savefig("hist_"+str(sigma)+".png")
	# plt.show()
	plt.hist(normalized.flatten(),201)
	plt.savefig("hist_"+str(sigma)+".png")
	plt.show()

	# save images
	normalized = ((normalized-np.min(normalized))/(np.max(normalized)-np.min(normalized)))*255
	cv2.imwrite("luminance_"+str(sigma)+".png",luminance)
	cv2.imwrite("normalized_"+str(sigma)+".png",normalized)

	while True:
		cv2.imshow("window",luminance)
		cv2.imshow("norm",normalized)

		# quit when click 'q' on keyboard
		k = cv2.waitKey(1)

		if k == ord('q'):
			cv2.destroyAllWindows()
			break