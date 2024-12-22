import cv2
import numpy as np


# Load stereo images
left_img = cv2.imread(r"C:\Users\drago\Downloads\checkerboard_pi_images\l.jpg", 0)
right_img = cv2.imread(r"C:\Users\drago\Downloads\checkerboard_pi_images\r.jpg", 0)
# Initialize stereo SGBM matcher
stereo = cv2.StereoSGBM_create(minDisparity=0,
                                numDisparities=16,
                                blockSize=5)
# Compute the disparity map
disparity = stereo.compute(left_img, right_img)
# Normalize the disparity map
disparity_normalized = cv2.normalize(src=disparity,
                                     dst=None,
                                     beta=0,
                                     alpha=255,
                                     norm_type=cv2.NORM_MINMAX)
# Convert to 8-bit image
disparity_normalized = np.uint8(disparity_normalized)
# Display the depth map
cv2.imshow('Depth Map', disparity_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()