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



if __name__ == "__main__":
	img1_path = "other-color-twoframes\\other-data\\RubberWhale\\frame10.png"
	img2_path = "other-color-twoframes\\other-data\\RubberWhale\\frame11.png"
	flo_path = "other-gt-flow\\other-gt-flow\\RubberWhale\\flow10.flo"
	flo_field = read(flo_path)
	img1 = cv2.imread(img1_path)
	img2 = cv2.imread(img2_path)

	# print((flo_field > 1) * (flo_field < 1e9))
	# exit()
	print(flo_field.shape)
	for row in range(0,flo_field.shape[0],8):
		for col in range(0,flo_field.shape[1],8):
			if flo_field[row,col,0] > 1e9:# or flo_field[row,col,0] < 1:
				continue
			else:
				w = int(flo_field[row,col,0])
				h = int(flo_field[row,col,1])
			start_point = (col, row) 
			end_point = (col+w, row+h) 
			color = (255, 255, 255) 
			thickness = 1
			# print(row,col)
			img2 = cv2.arrowedLine(img2, start_point, end_point, color, thickness) 
			img1 = cv2.arrowedLine(img1, start_point, start_point, color, thickness) 
	# print(col)
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
	# print(read(flo_path).shape)