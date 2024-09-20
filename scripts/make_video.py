import os

import cv2
from natsort import natsorted

# Directory where the images are stored
image_folder = "samples/bspline"  # Replace with the folder containing your images
output_video = "output_video.avi"  # Output video file name

# Get the list of images
images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
images = natsorted(images)
# images.sort()  # Sort the images to maintain the sequence
# exit()

# Read the first image to get the size (height, width) for the video
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"XVID")  # 'XVID' is a common codec for .avi files
video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

# Loop over all images and write them to the video
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)  # Write the frame to the video

# Release the VideoWriter object
video.release()

print(f"Video created and saved as {output_video}")
