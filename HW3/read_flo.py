import numpy as np
import os
import cv2

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
			end_point = (col+w, row+h) 
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

	# iterate over each block
	for row in block_idxs_h:
		for col in block_idxs_w:
			# get the search patch coordinates
			search_patch_tlx = col - block_dim//2 if col - block_dim//2 > 0 else 0
			search_patch_tly = row - block_dim//2 if row - block_dim//2 > 0 else 0
			search_patch_brx = col + block_dim//2 if col + block_dim//2 < w else w
			search_patch_bry = row + block_dim//2 if row + block_dim//2 < h else h
			search_patch = img1[search_patch_tly:search_patch_bry,search_patch_tlx:search_patch_brx]
			
			start_point = (col, row) 
			end_point = (col, row) 
			color = (255, 255, 255) 
			thickness = 1

			img1 = cv2.arrowedLine(img1, start_point, end_point, color, thickness)
			if row == 16*6 and col == 16*6:
				overlay = img1.copy()
				cv2.rectangle(overlay, (search_patch_tlx,search_patch_tly), (search_patch_brx,search_patch_bry), (0,0,255), -1)
				img1 = cv2.addWeighted(overlay, 0.2, img1, 1 - 0.2, 0)
	while True:
		cv2.imshow("window",img1)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if __name__ == "__main__":
	example = "Urban2"
	img1_path = "other-color-twoframes\\other-data\\" + example + "\\frame10.png"
	img2_path = "other-color-twoframes\\other-data\\" + example + "\\frame11.png"
	flo_path = "other-gt-flow\\other-gt-flow\\" + example + "\\flow10.flo"

	flo_field = read(flo_path)
	img1 = cv2.imread(img1_path)
	img2 = cv2.imread(img2_path)

	# viz_flo(flo_field,img1,img2)
	estimate_flow(img1,img2)
