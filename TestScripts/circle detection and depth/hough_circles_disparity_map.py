import random
import cv2
import numpy as np

#camera_matrix_1 = np.array([[801.55383955, 0, 305.58129421],[0, 800.9800178, 195.84030218],[0, 0, 1]])
camera_matrix_1 = np.array([[792.01009571, 0, 304.46152521],[0, 791.66246932, 192.88226236],[0, 0, 1]])
camera_matrix_2 = np.array([[792.01009571, 0, 304.46152521],[0, 791.66246932, 192.88226236],[0, 0, 1]])

dist_coeffs_1 = np.array([[7.52262753e-03, -3.37290154e-01, -3.58988068e-03, -4.36893114e-04, 1.95011206e+00]])
dist_coeffs_2 = np.array([[2.39409750e-02, -5.91783464e-01,  -2.17929649e-03, -1.15826319e-03, 3.99185295e+00]])

R1 = np.array([[0.97673312, -0.09184529, 0.19379589], [0.09166191, 0.99574065, 0.00993247], [-0.19388269, 0.00806233, 0.98099159]])
R2 = np.array([[0.97684858, -0.09261012, 0.19284765], [0.09279253, 0.99565247, 0.00810615], [-0.19275995, 0.00997634, 0.98119523]])
P1 = np.array([[7.96321244e+02, 0, 1.20930510e+02, -8.07371964e+03], [0, 7.96321244e+02, 1.82920061e+02, 0], [0, 0, 1, 0]])
P2 = np.array([[7.96321244e+02, 0, 1.20930510e+02, -8.07371964e+03], [0, 7.96321244e+02, 1.82920061e+02, 0], [0, 0, 1, 0]])

map1x, map1y = cv2.initUndistortRectifyMap(camera_matrix_1, dist_coeffs_1, R1, P1, (640, 360), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(camera_matrix_2, dist_coeffs_2, R2, P2, (640, 360), cv2.CV_32FC1)

img_left = cv2.imread(r"C:\Users\drago\Downloads\small_calib_screw_detect\7_l.jpg")
img_right = cv2.imread(r"C:\Users\drago\Downloads\small_calib_screw_detect\7_r.jpg")

rectified_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
rectified_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# Apply a blur to reduce noise
gray_blurred_l = cv2.medianBlur(rectified_l, 5)
gray_blurred_r = cv2.medianBlur(rectified_r, 5)

canny_gray_l = cv2.Canny(gray_blurred_l, 50, 30)
canny_gray_r = cv2.Canny(gray_blurred_r, 50, 30)


# Detect circles using Hough Circle Transform
circlesL = cv2.HoughCircles(gray_blurred_l, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                            param1=50, param2=20, minRadius=2, maxRadius=30)
circlesR = cv2.HoughCircles(gray_blurred_r, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                            param1=50, param2=20, minRadius=2, maxRadius=30)

class CircleManager():
    def __init__(self, image_l, image_r, circles_l, circles_r):
        self.circle_l_last_click = None
        self.circle_r_last_click = None

        self.image_l = image_l
        self.image_r = image_r
        self.annot_l = self.draw_circles(image_l, circles_l[0])
        self.annot_r = self.draw_circles(image_r, circles_r[0])

        self.circles_l = circles_l[0]
        self.circles_r = circles_r[0]
    
    # Function to draw detected circles on an image
    def draw_circles(self, image, circles):
        annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if circles is not None:
            circles = np.round(circles).astype("int")
            for i, (x, y, r) in enumerate(circles):
                if self.circle_l_last_click == i or self.circle_r_last_click == i:
                    colour = (0, 0, 255)
                else:
                    colour = (0, 255, 0)
                cv2.circle(annotated_image, (x, y), r, colour, 2)
        return annotated_image

    # Mouse callback function for clicking events
    def on_click_image_l(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if isinstance(self.circle_l_last_click, int):
                (cx, cy, r) = self.circles_l[self.circle_l_last_click]
                # If we click inside the same circle, move the centre to the new location
                # Otherwise, delete it
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    self.circles_l[self.circle_l_last_click] = (x, y, r)
                    self.circle_l_last_click = None
                else:
                    self.circles_l = np.delete(self.circles_l, self.circle_l_last_click, 0)
                    self.circle_l_last_click = None
            
            # Looking for the click to be inside the same circle as last time to indicate move command
            # Looking for a click outside all screws to indicate delete command
            elif self.circle_l_last_click==None:
                for i, (cx, cy, r) in enumerate(self.circles_l):
                    if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                        self.circle_l_last_click = i
                        break
            
            self.annot_l = self.draw_circles(self.image_l, self.circles_l)
            cv2.imshow("Image_left", self.annot_l)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Mouse callback function for clicking events
    def on_click_image_r(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if isinstance(self.circle_r_last_click, int):
                (cx, cy, r) = self.circles_r[self.circle_r_last_click]
                # If we click inside the same circle, move the centre to the new location
                # Otherwise, delete it
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    self.circles_r[self.circle_r_last_click] = (x, y, r)
                    self.circle_r_last_click = None
                else:
                    self.circles_r = np.delete(self.circles_r, self.circle_r_last_click, 0)
                    self.circle_r_last_click = None
            
            # Looking for the click to be inside the same circle as last time to indicate move command
            # Looking for a click outside all screws to indicate delete command
            elif self.circle_r_last_click==None:
                for i, (cx, cy, r) in enumerate(self.circles_r):
                    if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                        self.circle_r_last_click = i
                        break
            
            self.annot_r = self.draw_circles(self.image_r, self.circles_r)
            cv2.imshow("Image_right", self.annot_r)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

manager = CircleManager(rectified_l, rectified_r, circlesL, circlesR)

# # Show the images with detected circles
cv2.imshow("Image_left", manager.annot_l)
cv2.imshow("Image_right", manager.annot_r)
cv2.setMouseCallback("Image_left", manager.on_click_image_l)
cv2.setMouseCallback("Image_right", manager.on_click_image_r)

cv2.waitKey(0)
cv2.destroyAllWindows()

circlesL, circlesR = manager.circles_l, manager.circles_r

# Define an offset variable
offset_x = 10 # Adjust this value as needed

x_range = 70
y_range = 20

if circlesL is not None and circlesR is not None:
    circlesL = np.round(circlesL).astype("int")
    circlesR = np.round(circlesR).astype("int")
    
    # List to store matched circle pairs and their disparities
    matched_circles = []
    
    # Match circles between left and right images
    for (xL, yL, rL) in circlesL:
        for (xR, yR, rR) in circlesR:
            # Check if the circles are approximately at the same y coordinate
            # TODO: Rein in so screws correctly match
            if abs(yL - yR) < y_range:  # Adjust tolerance as needed
                if abs((xL + offset_x) - xR) < x_range:
                    disparity = abs(xL - xR)
                    matched_circles.append(((xL, yL, rL), (xR, yR, rR), disparity))

    rectified_l = cv2.cvtColor(rectified_l, cv2.COLOR_GRAY2BGR)
    rectified_r = cv2.cvtColor(rectified_r, cv2.COLOR_GRAY2BGR)

    # Draw and display the matched circles and their disparity
    for ((xL, yL, rL), (xR, yR, rR), disparity) in matched_circles:
        colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.circle(rectified_l, (xL, yL), rL, colour, 2)
        cv2.circle(rectified_r, (xR, yR), rR, colour, 2)
        print(f"Circle at ({xL}, {yL}) in left image matches with ({xR}, {yR}) in right image. Disparity: {disparity}")
    
    cv2.imshow("Image_left", rectified_l)
    cv2.imshow("Image_right", rectified_r)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No circles were detected in one or both images.")

cv2.imwrite(r"C:\Users\drago\Downloads\small_calib_screw_detect\7_l_annot_v2.jpg", rectified_l)
cv2.imwrite(r"C:\Users\drago\Downloads\small_calib_screw_detect\7_r_annot_v2.jpg", rectified_r)