import cv2

# Replace with your RTSP stream URL
url = "http://192.168.137.138:8000/stream.mjpg"

# Create a VideoCapture object
cap = cv2.VideoCapture(url)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# When everything is done, release the video capture object
cap.release()

# Close all the frames
cv2.destroyAllWindows()
