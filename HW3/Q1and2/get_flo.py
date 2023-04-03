import numpy as np
import os
import cv2
from tqdm import tqdm


'''
I took this read() code from https://github.com/Johswald/flow-code-python/blob/master/readFlowFile.py
to read the .flo files.
'''
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
	data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
	flow = np.resize(data, (int(h[0]), int(w[0]), 2))	
	f.close()

	return flow


'''Visualize the motion vectors'''
def viz_flo(flo_field,img1,img2,sample_img=16,sample_field=16,color=(255, 255, 255)):
	for row in range(0,flo_field.shape[0],sample_field):
		for col in range(0,flo_field.shape[1],sample_field):
			if flo_field[row,col,0] > 1e9:
				w = 0
				h = 0
			else:
				w = int(flo_field[row,col,0])
				h = int(flo_field[row,col,1])
			start_point = (col*sample_img//sample_field, row*sample_img//sample_field) 
			col_new = col*sample_img//sample_field + w
			row_new = row*sample_img//sample_field + h
			
			if col_new < 0:
				col_new = 0
			elif col_new > img1.shape[1]:
				col_new = img1.shape[1]
			if row_new < 0:
				row_new = 0
			elif row_new > img1.shape[0]:
				row_new = img1.shape[0]
			end_point = (col_new, row_new) 
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

	return img2


''' estimate the flow'''
def estimate_flow(img1,img2,method="brute_force",block_dim=16,search_dim=48):
	total_count = 0

	# get the top left corner of each block based on the desired block size
	h,w = img1.shape[0],img1.shape[1]
	block_idxs_h = np.arange(h)[::block_dim]
	block_idxs_w = np.arange(w)[::block_dim]
	block_idxs_h = block_idxs_h if block_idxs_h[-1]+block_dim <= h else block_idxs_h[:-1]
	block_idxs_w = block_idxs_w if block_idxs_w[-1]+block_dim <= w else block_idxs_w[:-1]

	# store estimated flo vectors
	flo_estimate = np.zeros((len(block_idxs_h),len(block_idxs_w),2))

	
	img2_disp = img2.copy()
	cv2.imshow('window',img2_disp)
	
	# iterate over each block
	for row_i,row in enumerate(tqdm(block_idxs_h)):
		for col_i,col in enumerate(block_idxs_w):
			
			# get the search patch coordinates, if these coordinate go beyond the boundary, the corresponding block is ignored
			search_patch_tlx = col - search_dim//2
			search_patch_tly = row - search_dim//2
			search_patch_brx = col + search_dim//2
			search_patch_bry = row + search_dim//2
			# print(f"-------------\n row:{row},col:{col},tlx:{search_patch_tlx},tly:{search_patch_tly},brx:{search_patch_brx},bry:{search_patch_bry}")
			
			# the refernce block from the last frame for which we want to find the motion vectors
			last_block = img1[row:row+block_dim,col:col+block_dim]

			# we use sum of absolute differences to compare (l1 norm)
			l1 = np.ones((search_dim+1,search_dim+1))*1e9

			# search for the best match
			if method == "brute_force":
				per_block = 0
				for i,sub_row in enumerate(range(search_patch_tly,search_patch_bry+1)):
					for j,sub_col in enumerate(range(search_patch_tlx,search_patch_brx+1)):
						# print(f"subrow:{sub_row},subcol:{sub_col},i:{i},j:{j}")
						# if we go outside the frame, continue to next block comparison
						if sub_col < 0 or sub_row < 0 or \
						sub_col + block_dim > w or sub_row + block_dim > h:
							continue
						pred_block = img2[sub_row:sub_row+block_dim,sub_col:sub_col+block_dim]
						try:
							# print("-- ",i,j,sub_row,sub_col,search_patch_tlx,search_patch_tly,search_patch_brx,search_patch_bry)
							l1[i,j] = np.sum(np.abs(pred_block.astype(int)-last_block.astype(int)))
							total_count += 1
						except Exception as e:
							print(i,j,e)
							exit()
				# now we get the index of the best matching block
				closest = np.argmin(l1)
				row_num = (closest // (search_dim+1))
				col_num = (closest % (search_dim+1))

				# convert to relative offset (e.g. (0,0) --> (-24,-24))
				y_rel = row_num-search_dim//2
				x_rel = col_num-search_dim//2

			elif method == "log":
				l1_sub = np.ones(5)*1e9
				l1_final = np.ones(9)*1e9

				# initialize the search edge to half the distance to the search space boundary
				edge = search_dim // 4
				curr_center = np.array([row,col])
				# img2_disp = img2.copy()
				# cv2.rectangle(img2_disp,(curr_center[1]-search_dim//2,curr_center[0]-search_dim//2),(curr_center[1]+search_dim//2,curr_center[0]+search_dim//2),(0,255,0),1)
				per_block = 0
				
				# continue the search until reach the base case
				while edge > 1:
					# the search directions
					dirs = np.array([[0,0],[0,-edge],[-edge,0],[0,edge],[edge,0]])

					# after the first iteration we can reuse the center point
					start = 0 if edge == search_dim // 4 else 1
					for i in range(start,5):
						if ((curr_center + dirs[i]) < 0).any() or ((curr_center[0] + dirs[i][0] + block_dim) > h) or ((curr_center[1] + dirs[i][1] + block_dim) > w):
							continue
						pred_block = img2[curr_center[0]+dirs[i][0]:curr_center[0]+dirs[i][0]+block_dim,curr_center[1]+dirs[i][1]:curr_center[1]+dirs[i][1]+block_dim]
						l1_sub[i] = np.sum(np.abs(pred_block.astype(int)-last_block.astype(int)))
						total_count += 1
						per_block += 1
						# if row > 300:
						# cv2.rectangle(img2_disp,((curr_center+dirs[i])[1],(curr_center+dirs[i])[0]),((curr_center+dirs[i])[1]+block_dim,(curr_center+dirs[i])[0]+block_dim),(0,0,255),1)
						# cv2.imshow('window',img2_disp)
						# if cv2.waitKey(150) & 0xFF == ord('q'):
						# 	break
						# print(f"curr: {curr_center}, edge: {edge}, dir: {dirs[i]}, comp: {curr_center+dirs[i]}, diff: {l1_sub[i]}, block:{curr_center[0]+dirs[i][0]}->{curr_center[0]+dirs[i][0]+block_dim},{curr_center[1]+dirs[i][1]}->{curr_center[1]+dirs[i][1]+block_dim}")
					closest = np.argmin(l1_sub)
					curr_center += dirs[closest]
					edge = edge // 2
					l1_sub[0] = l1_sub[closest]
					l1_sub[1:] = 1e9
					
				# after we reach the base case, do a final fine-grained search
				dirs = np.array([[0,0],[0,-1],[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1]])
				l1_final[0] = l1_sub[closest]
				for i in range(1,9):
					per_block += 1
					if ((curr_center + dirs[i]) < 0).any() or ((curr_center[0] + dirs[i][0] + block_dim) > h) or ((curr_center[1] + dirs[i][1] + block_dim) > w):
							continue
					pred_block = img2[curr_center[0]+dirs[i][0]:curr_center[0]+dirs[i][0]+block_dim,curr_center[1]+dirs[i][1]:curr_center[1]+dirs[i][1]+block_dim]
					l1_final[i] = np.sum(np.abs(pred_block.astype(int)-last_block.astype(int)))
					total_count += 1
					# print(f"curr: {curr_center}, edge: {edge}, dir: {dirs[i]}, comp: {curr_center+dirs[i]}, diff: {l1_final[i]}, block:{curr_center[0]+dirs[i][0]}->{curr_center[0]+dirs[i][0]+block_dim},{curr_center[1]+dirs[i][1]}->{curr_center[1]+dirs[i][1]+block_dim}")
				closest = np.argmin(l1_final)
				curr_center += dirs[closest]
				x_rel = curr_center[1] - col
				y_rel = curr_center[0] - row
			
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
		# exit()
	print("Total Comparisons:",total_count)
	return flo_estimate

if __name__ == "__main__":
	example = "Urban2"
	img1_path = "other-color-twoframes\\other-data\\" + example + "\\frame10.png"
	img2_path = "other-color-twoframes\\other-data\\" + example + "\\frame11.png"
	flo_path = "other-gt-flow\\other-gt-flow\\" + example + "\\flow10.flo"

	flo_field = read(flo_path)
	img1 = cv2.imread(img1_path)
	img2 = cv2.imread(img2_path)

	flow_img_gt = viz_flo(flo_field,img1,img2.copy(),color=(0,255,0))
	cv2.imwrite("gt.png",flow_img_gt)

	ff = estimate_flow(img1,img2,method="brute_force")
	flow_img_est = viz_flo(ff,img1,img2,sample_img=16,sample_field=1)
	cv2.imwrite("estimate.png",flow_img_est)

	img = cv2.addWeighted(flow_img_gt, 0.5, flow_img_est, 1 - 0.5, 0)
	cv2.imwrite("overlayed.png",img)

	print("MSE:",np.mean(np.square(ff - flo_field[::16,::16])))
