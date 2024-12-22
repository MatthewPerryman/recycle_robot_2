import cv2
import numpy as np

camera_matrix_1 = np.array([[801.55383955, 0, 305.58129421],[0, 800.9800178, 195.84030218],[0, 0, 1]])
camera_matrix_2 = np.array([[792.01009571, 0, 304.46152521],[0, 791.66246932, 192.88226236],[0, 0, 1]])

dist_coeffs_1 = np.array([[7.52262753e-03, -3.37290154e-01, -3.58988068e-03, -4.36893114e-04, 1.95011206e+00]])
dist_coeffs_2 = np.array([[2.39409750e-02, -5.91783464e-01,  -2.17929649e-03, -1.15826319e-03, 3.99185295e+00]])

R1 = np.array([[0.97673312, -0.09184529, 0.19379589], [0.09166191, 0.99574065, 0.00993247], [-0.19388269, 0.00806233, 0.98099159]])
R2 = np.array([[0.97684858, -0.09261012, 0.19284765], [0.09279253, 0.99565247, 0.00810615], [-0.19275995, 0.00997634, 0.98119523]])
P1 = np.array([[7.96321244e+02, 0, 1.20930510e+02, -8.07371964e+03], [0, 7.96321244e+02, 1.82920061e+02, 0], [0, 0, 1, 0]])
P2 = np.array([[7.96321244e+02, 0, 1.20930510e+02, -8.07371964e+03], [0, 7.96321244e+02, 1.82920061e+02, 0], [0, 0, 1, 0]])

map1x, map1y = cv2.initUndistortRectifyMap(camera_matrix_1, dist_coeffs_1, R1, P1, (640, 360), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(camera_matrix_2, dist_coeffs_2, R2, P2, (640, 360), cv2.CV_32FC1)

img_left = cv2.imread(r"C:\Users\drago\Downloads\small_calib_screw_detect\2_l.jpg")
img_right = cv2.imread(r"C:\Users\drago\Downloads\small_calib_screw_detect\2_r.jpg")

# Rectify the images
rectified_l = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
rectified_r = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)

rectified_l = cv2.cvtColor(rectified_l, cv2.COLOR_BGR2GRAY)
rectified_r = cv2.cvtColor(rectified_r, cv2.COLOR_BGR2GRAY)

# Create a window to display the images
cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Disparity', 800, 600)

# Create trackbars for parameter tuning
def nothing(x):
    pass

cv2.createTrackbar('numDisparities', 'Disparity', 1, 17, nothing)  # Needs to be multiple of 16
cv2.createTrackbar('blockSize', 'Disparity', 5, 50, nothing)       # Needs to be odd number >=5
cv2.createTrackbar('uniquenessRatio', 'Disparity', 5, 15, nothing)
cv2.createTrackbar('speckleWindowSize', 'Disparity', 0, 200, nothing)
cv2.createTrackbar('speckleRange', 'Disparity', 0, 2, nothing)
cv2.createTrackbar('disp12MaxDiff', 'Disparity', 1, 25, nothing)

while True:
    # Get current positions of trackbars
    numDisparities = cv2.getTrackbarPos('numDisparities', 'Disparity') * 16
    blockSize = cv2.getTrackbarPos('blockSize', 'Disparity')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'Disparity')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'Disparity')
    speckleRange = cv2.getTrackbarPos('speckleRange', 'Disparity')
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'Disparity')

    # Ensure blockSize is odd and >= 5
    if blockSize % 2 == 0:
        blockSize += 1
    if blockSize < 5:
        blockSize = 5

    # Initialize the stereo block matcher
    #stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    stereo = cv2.StereoSGBM_create( minDisparity=0, numDisparities=numDisparities, blockSize=blockSize,
                                   uniquenessRatio=uniquenessRatio, speckleWindowSize=speckleWindowSize,
                                   speckleRange=speckleRange, disp12MaxDiff=disp12MaxDiff)

    # Compute the disparity image
    disparity = stereo.compute(rectified_l, rectified_r).astype(np.float32) / 16.0

    # Normalize the image for display
    disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min())
    disparity = (disparity * 255).astype(np.uint8)

    # Display the result
    cv2.imshow('Disparity', disparity)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()