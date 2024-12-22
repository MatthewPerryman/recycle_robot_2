import os
import cv2
import numpy as np
import random

def show_image_with_click_coordinates(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Create a named window
    cv2.namedWindow('Image')

    # Initialize an empty list to store clicked coordinates and colors
    clicked_coordinates = []
    dot_colors = []

    def get_coordinates(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at (x, y): ({x}, {y})")
            clicked_coordinates.append((x, y))
            # Generate a random color for the dot
            dot_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            dot_colors.append(dot_color)
            cv2.circle(img, (x, y), 5, dot_color, -1)  # Overlay colored dot
            cv2.imshow('Image', img)

    # Bind the callback function to the window
    cv2.setMouseCallback('Image', get_coordinates)

    res = img.shape[0]//1.5
    img = cv2.resize(img, (int(img.shape[1]//1.5), int(img.shape[0]//1.5)))
    # Display the image
    cv2.imshow('Image', img)

    # Wait for 'q' key to exit
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Save the annotated image
    file = image_path.split('.')[0] + '_annot.jpg'
    cv2.imwrite(file, img)
    print(f"Annotated image saved as {file}")

    # Clean up
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = r"C:\Users\drago\Downloads\checkerboard_pi_images\test\localise_hr_test_2.jpg"  # Replace with your image file path
    show_image_with_click_coordinates(image_path)

