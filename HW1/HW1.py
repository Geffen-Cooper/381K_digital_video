# import libraries
import cv2
import numpy as np


# this function reformats into YUV422 and YUV420
def sample_YUV(frame,format):
    # print("before:\n",frame[:4,:4,:])
    if format == '422':
        # this samples every other element and
        # does nearest neighbor interpolation in one shot
        frame[:,1::2,1:] = frame[:,0::2,1:]
    elif format == '420':
        # this samples every other element (row and col) and
        # does nearest neighbor interpolation in one shot
        frame[:,1::2,1:] = frame[:,::2,1:]
        frame[1::2,:,1:] = frame[::2,:,1:]
    # print("after:\n",frame[:4,:4,:])
    return frame


# this function reads a single frame from a file
def read_frame(video_file,w,h,output_format):
    # initialize frame
    num_pixels = w*h
    frame = np.zeros((h,w,3),np.uint8)

    # iterate over each pixel
    for pixel_idx in range(num_pixels):
        # each pixel is 24 bits (3 bytes)
        # YYYYYYYY UUUUUUUU VVVVVVVV
        pixel = video_file.read(3)

        # check if valid
        if pixel:
            # get the x,y coordinate of the pixel
            x = pixel_idx%w
            y = pixel_idx//w

            # get the YUV bytes
            Y = pixel[0]
            U = pixel[1]
            V = pixel[2]
            frame[y,x,:] = (Y,U,V)
        else:
            return None
    # convert the sampled version to BGR
    return cv2.cvtColor(sample_YUV(frame,output_format),cv2.COLOR_YUV2BGR)
        
# this function calls the read frame function for the whole video
def read_video(video_path,w,h,output_format):
    # open the video file
    video_file = open(video_path,"rb")

    # then grab each frame based on w x h
    frame_count = 0
    next_frame = read_frame(video_file,w,h,output_format)
    last_frame = next_frame
    while next_frame is not None:
        frame_count += 1
        last_frame = next_frame
        next_frame = read_frame(video_file,w,h,output_format)
    return (frame_count,last_frame)

if __name__ == "__main__":
    # read the video to get the number of frames and the last frame
    format = '444' # '422', '420'
    frame_count, last_frame = read_video("tulips_yuv444_prog_packed_qcif.yuv",\
                                         176,144,format)

    # display results
    print("frame count:",frame_count)
    cv2.imshow("last frame",last_frame)
    cv2.waitKey(0)
    cv2.imwrite(format+'.png',last_frame)