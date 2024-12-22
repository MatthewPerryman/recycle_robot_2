import cv2
import numpy as np

img_l = cv2.imread(r"C:\Users\drago\Downloads\small_calib_screw_detect\2_l.jpg")
img_r = cv2.imread(r"C:\Users\drago\Downloads\small_calib_screw_detect\2_r.jpg")

# Apply the sobel operator
#img = cv2.GaussianBlur(img, (3, 3), 0)
gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray l', gray_l)
cv2.waitKey(0)
gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray r', gray_r)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Apply a blur to reduce noise
gray_blurred_l = cv2.medianBlur(gray_l, 5)
gray_blurred_r = cv2.medianBlur(gray_r, 5)

canny_gray_l = cv2.Canny(gray_blurred_l, 50, 30)
canny_gray_r = cv2.Canny(gray_blurred_r, 50, 30)
cv2.imshow('canny_gray_r', canny_gray_l)
cv2.waitKey(0)
cv2.imshow('canny_gray_r', canny_gray_r)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Detect circles using HoughCircles
circles_l = cv2.HoughCircles(gray_blurred_l, 
                           cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                           param1=50, param2=30, minRadius=5, maxRadius=20)

circles_r = cv2.HoughCircles(gray_blurred_r, 
                           cv2.HOUGH_GRADIENT, dp=1, minDist=5,
                           param1=50, param2=30, minRadius=5, maxRadius=20)

# If some circles are detected, overlay them on the image
if circles_l is not None:
    circles_l = np.uint16(np.around(circles_l))
    for i in circles_l[0, :]:
        # Draw the outer circle
        cv2.circle(img_l, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw the center of the circle
        cv2.circle(img_l, (i[0], i[1]), 2, (0, 0, 255), 3)

if circles_r is not None:
    circles_r = np.uint16(np.around(circles_r))
    for i in circles_r[0, :]:
        # Draw the outer circle
        cv2.circle(img_r, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw the center of the circle
        cv2.circle(img_r, (i[0], i[1]), 2, (0, 0, 255), 3)

# Display the result
cv2.imshow('Detected Circles Left', img_l)
cv2.waitKey(0)

cv2.imshow('Detected Circles Right', img_r)
cv2.waitKey(0)
cv2.destroyAllWindows()

exit()
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
    
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

#grad = cv2.resize(grad, (int(grad.shape[1]//3), int(grad.shape[0]//3)))

cv2.imshow('image', grad)
cv2.waitKey(0)