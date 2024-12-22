import cv2
import numpy as np
from matplotlib import pyplot as plt

imgL = cv2.imread(r"C:\Users\drago\Downloads\checkerboard_pi_images\test\laptop_2.jpg", cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread(r"C:\Users\drago\Downloads\checkerboard_pi_images\test\laptop_1.jpg", cv2.IMREAD_GRAYSCALE)

imgL = cv2.blur(imgL, (10, 10))
imgR = cv2.blur(imgR, (10, 10))

plt.imshow(imgL,'gray')
plt.show()
plt.imshow(imgR,'gray')
plt.show()

stereo = cv2.StereoBM_create( numDisparities=16, blockSize=21)
stereo.setTextureThreshold(30)
stereo.setSpeckleRange(1000)
stereo.setMinDisparity(500)
stereo.setNumDisparities(960)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()

print(stereo.getBlockSize())
 
print(stereo.getDisp12MaxDiff())
 
print(stereo.getMinDisparity())
 
print(stereo.getNumDisparities())
 
print(stereo.getSpeckleRange())
 
print(stereo.getSpeckleWindowSize())
 
# print(stereo.setBlockSize (int blockSize)=0
 
# print(stereo.setDisp12MaxDiff (int disp12MaxDiff)=0
 
# print(stereo.setMinDisparity (int minDisparity)=0
 
# print(stereo.setNumDisparities (int numDisparities)=0
 
# print(stereo.setSpeckleRange (int speckleRange)=0
 
# print(stereo.setSpeckleWindowSize (int speckleWindowSize)=0