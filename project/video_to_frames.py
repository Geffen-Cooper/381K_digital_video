import cv2

# Open the video file
video_path = "example.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# Loop through each frame in the video
frame_number = 0
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if we have reached the end of the video
    if not ret:
        break

    # Save the frame as an image file
    frame_filename = f"frame_{frame_number:04d}.png"  # Change the file format to suit your needs
    cv2.imwrite(frame_filename, frame)

    # Print progress
    print(f"Saved frame {frame_filename}")

    # Increment frame number
    frame_number += 1

# Release the video file and close all windows
cap.release()
cv2.destroyAllWindows()