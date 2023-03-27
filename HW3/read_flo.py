import numpy as np
import os
import cv2
from tqdm import tqdm
TAG_FLOAT = 202021.25

def read(file):

	assert type(file) is str, "file is not str %r" % str(file)
	assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
	assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
	f = open(file,'rb')
	flo_number = np.fromfile(f, np.float32, count=1)[0]
	assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
	w = np.fromfile(f, np.int32, count=1)
	h = np.fromfile(f, np.int32, count=1)
	#if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
	data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
	# Reshape data into 3D array (columns, rows, bands)
	flow = np.resize(data, (int(h[0]), int(w[0]), 2))	
	f.close()

	return flow

def viz_flo(flo_field,img1,img2,sample=16):
	for row in range(0,flo_field.shape[0],sample):
		for col in range(0,flo_field.shape[1],sample):
			if flo_field[row,col,0] > 1e9:
				w = 0
				h = 0
			else:
				w = int(flo_field[row,col,0])
				h = int(flo_field[row,col,1])
			start_point = (col, row) 
			col_new = col + w
			row_new = row + h
			if col_new < 0:
				col_new = 0
			if col_new > img1.shape[1]:
				col_new = img1.shape[1]
			if row_new < 0:
				row_new = 0
			if row_new > img1.shape[0]:
				row_new = img1.shape[0]
			end_point = (col_new, row_new) 
			color = (255, 255, 255) 
			thickness = 1

			img2 = cv2.arrowedLine(img2, start_point, end_point, color, thickness) 
			img1 = cv2.arrowedLine(img1, start_point, start_point, color, thickness) 
	
	next = True
	while True:
		if next:
			cv2.imshow("window",img1)
		else:
			cv2.imshow("window",img2)
		next = not next

		# quit when click 'q' on keyboard
		if cv2.waitKey(1000) & 0xFF == ord('q'):
			break

# estimate the flow
def estimate_flow(img1,img2,method="brute_force",block_dim=16,search_dim=48):
	# get the top left corner of each block based on the desired block size
	h,w = img1.shape[0],img1.shape[1]
	block_idxs_h = np.arange(h)[::block_dim]
	block_idxs_w = np.arange(w)[::block_dim]

	# store estimated flo vectors
	flo_estimate = np.zeros((len(block_idxs_h),len(block_idxs_w),2))

	# print(f"w:{w},h:{h},blocks_w:{len(block_idxs_w)},blocks_h:{len(block_idxs_h)}")

	# iterate over each block
	for row_i,row in enumerate(tqdm(block_idxs_h)):
		for col_i,col in enumerate(block_idxs_w):
			
			# get the search patch coordinates, make sure stay within boundaries
			search_patch_tlx = col - search_dim//2 #if col - search_dim//2 > 0 else 0
			search_patch_tly = row - search_dim//2 #if row - search_dim//2 > 0 else 0
			search_patch_brx = col + search_dim//2 #if col + search_dim//2 + block_dim < w else w - block_dim
			search_patch_bry = row + search_dim//2 #if row + search_dim//2 + block_dim < h else h - block_dim
			# print(f"-------------\n row:{row},col:{col},tlx:{search_patch_tlx},tly:{search_patch_tly},brx:{search_patch_brx},bry:{search_patch_bry}")
			
			# we compare against all blocks in the search batch of the next frame
			last_block = img1[row:row+block_dim,col:col+block_dim]

			# we use sum of absolute differences to compare (l1 norm)
			l1 = np.ones((search_dim+1,search_dim+1))*1e9
			for i,sub_row in enumerate(range(search_patch_tly,search_patch_bry+1)):
				for j,sub_col in enumerate(range(search_patch_tlx,search_patch_brx+1)):
					# print(f"subrow:{sub_row},subcol:{sub_col},i:{i},j:{j}")
					# if we go outside the frame, continue to next block comparison
					if sub_col < 0 or sub_row < 0 or \
					   sub_col + block_dim > w or sub_row + block_dim > h:
						# print("*")
						continue
					pred_block = img2[sub_row:sub_row+block_dim,sub_col:sub_col+block_dim]
					try:
						# print("-- ",i,j,sub_row,sub_col,search_patch_tlx,search_patch_tly,search_patch_brx,search_patch_bry)
						l1[i,j] = np.sum(np.abs(pred_block.astype(int)-last_block.astype(int)))
					except Exception as e:
						print(i,j,e)
						exit()
			
			# now we get the index of the best matching block
			closest = np.argmin(l1)
			row_num = (closest % (search_dim+1))
			col_num = (closest // (search_dim+1))

			# convert to relative coordinates (e.g. (0,0) --> (-24,-24))
			y_rel = row_num-search_dim//2
			x_rel = col_num-search_dim//2

			print(x_rel,y_rel)

			flo_estimate[row_i,col_i,0] = x_rel
			flo_estimate[row_i,col_i,1] = y_rel
			
			# start_point = (col, row) 
			# end_point = (col, row) 
			# color = (255, 255, 255) 
			# thickness = 1

			# img1 = cv2.arrowedLine(img1, start_point, end_point, color, thickness)
			# if row == 16*6 and col == 16*6:
			# 	overlay = img1.copy()
			# 	cv2.rectangle(overlay, (search_patch_tlx,search_patch_tly), (search_patch_brx,search_patch_bry), (0,0,255), -1)
			# 	img1 = cv2.addWeighted(overlay, 0.2, img1, 1 - 0.2, 0)
	# while True:
	# 	cv2.imshow("window",img1)
	# 	if cv2.waitKey(1) & 0xFF == ord('q'):
	# 		break
	return flo_estimate

if __name__ == "__main__":
	example = "Urban2"
	img1_path = "other-color-twoframes\\other-data\\" + example + "\\frame10.png"
	img2_path = "other-color-twoframes\\other-data\\" + example + "\\frame11.png"
	flo_path = "other-gt-flow\\other-gt-flow\\" + example + "\\flow10.flo"

	flo_field = read(flo_path)
	img1 = cv2.imread(img1_path)
	img2 = cv2.imread(img2_path)

	# viz_flo(flo_field,img1,img2)
	ff = estimate_flow(img1,img2)
	viz_flo(ff,img1,img2,sample=1)
