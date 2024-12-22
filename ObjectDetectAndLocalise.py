# Importing Required Modules
import cv2
from matplotlib import pyplot as plt
import requests
import numpy as np
from absl import app, logging_absl
from time import time
import math
import io
import json
from PIL import Image as Image, ImageDraw
import subprocess
import re
import torch
from ultralytics import YOLO
import os
import signal
from enum import Enum

#from Model.detect import Classifier
from utils import logging

class function_type(Enum):
	GATHER_DATA = 1
	DETECT_MULTIPLE_SCREWS = 2

server_address = "http://192.168.137.28:1024"
model_type = 'yolov8'

dataset_name = "test_dataset"
base_directory = f"Autogathered_Dataset/{dataset_name}/"
IMG_LABELS_FILE = base_directory + "img_labels.json"


Zd_depth = -135  # (mm) = 14 cm with negative because it is in the -z direction (up)
#Zd_depth = -100  # (mm) = 10 cm with negative because it is in the -z direction (up)
focal_length = 4.74	# (mm) = 0.474 cm with negative because it is in the -z direction (up)
# Assuming the camera is fixed at lens position: 5.6818181818 (lense is at the position that allows Zd=14cm to in focus)
# A rearrangement of 1/u + 1/v = 1/f
u = focal_length * abs(Zd_depth) / (focal_length + abs(Zd_depth))

# Vector from motor tip to camera
# At 0,0,-10, offset is 1 cm below camera
camera_offset = (0, 0, 10)

# Offset vector due to robot effector location being set to the suction cup.
# This difference is between the suction cup and the camera
robot_head_to_camera = (0.9, 0, 1.8)

# Note: all coordinates are x, y
image_shape = (480, 640, 3)
m_frame_distance = (0, 0, -10)

# Find the center pixel in the image
image_center = (image_shape[1] / 2, image_shape[0] / 2)

# Based upon datasheet https://cdn.sparkfun.com/datasheets/Dev/RaspberryPi/ov5647_full.pdf#:~:text=The%20OV5647%20is
# %20a%20low%20voltage%2C%20high%20performance%2C,the%20serial%20camera%20control%20bus%20or%20MIPI%20interface.
pixel_size = 0.0014  # (mm) = 1.4 micrometers

robot_default_location = {'Xd': 200, 'Yd': 0, 'Zd': 150}

class Comms():
	def set_position(self, vector_target):
		# Call API to move robot
		status = requests.post(server_address + "/set_position/", data=bytes(json.dumps(vector_target), 'utf-8'))

		if json.loads(status.content)['response'] == True:
			print("Move successful")
			print("Robot Location: {}".format(vector_target))
			logging.write_log("client", "Robot moved to {}".format(vector_target))
		else:
			print("Move Out of Bounds")
	
	def get_photo(self):
		img_request = requests.get(server_address + "/get_photo")
		logging.write_log("client", "Received Image from Server")

		# Validate image request response
		if img_request.status_code != requests.codes.ok:
			print("Error: {}".format(img_request.status_code))
			return False

		# Read numpy array bytes
		np_zfile = np.load(io.BytesIO(img_request.content))

		self.image = np_zfile['arr_0']

		return True
		
	def move_by_vector(self, destination):
		return json.loads(requests.post(server_address + "/move_by_vector/",
										data=bytes(json.dumps(destination), 'utf-8')))['response']
	
	def get_position(self):
		return requests.get(server_address + "/get_position/").content
	
	def get_simple_photo(self):
		return requests.get(server_address + "/get_simple_photo")
	
	def get_wrist_angle(self):
		return requests.get(server_address + "/get_wrist_angle")


class Client():
	def __connect__(self):
		try:
			output = subprocess.check_output(["ping", "-4", "raspberrypi.local"])
			output = output.decode("utf-8")
			match = re.search(r"\[([^\]]+)\]", output)
			print(match.group(1))
			if match:
				ip_address = match.group(1)
				if ip_address:
					global server_address
					server_address = f"http://{ip_address}:1024"
					print(f"Raspberry Pi found at IP address: {ip_address}")
					return True
				else:
					print("Raspberry Pi not found")
					return False
			return False
		except subprocess.CalledProcessError:
			return False

	def __init__(self):
		# Connect to the raspberry pi and load the machine learning model
		try:
			if self.__connect__():
				print(torch.cuda.device_count())

				if model_type == 'yolov8':
					# Load trained model from weights file
					self.model = YOLO('yolov8_model/2-yolo-best.pt')
					self.model.predict([np.zeros((640, 480, 3), dtype=np.uint8)])

				self.comms = Comms()
			else:
				exit()
		except subprocess.CalledProcessError:
			return exit()

	# Stream images from the camera and detect any screws present in the image
	def detect_screws_in_stream(self):
		t1, t2, c1, c2 = 0, 0, 0, 0
		while True:
			t1 = time()
			# Exception Handling for invalid requests
			try:
				c1 = time()

				# Verifying whether the specified URL exist or not
				self.image = self.get_simple_photo()

				# If no error from getting function
				if type(self.image) is np.ndarray:
					c2 = time()
					t2 = time()

					# Display the labelled image with a delay of 1 millisecond (minimum delay)
					self.model.detect()

					self.show_image()
				else:
					print("Get photo failed")

				logging_absl.info('Call API time: {}'.format(c2 - c1))
				logging_absl.info('Full loop time: {}'.format(t2 - t1))
			except Exception as e:
				print(str(e))


	def move_robot(self):
		print("Print ctrl+c to quite from here")
		while True:
			# Get user input for the command separated location
			target = input("Target location <x,y,z>: ")
			vector_target = [float(axis) for axis in target.split(',')]

			print("Trying to move to: {},{},{}".format(vector_target[0], vector_target[1], vector_target[2]))

			vector_target = {'Xd': vector_target[0], 'Yd': vector_target[1], 'Zd': vector_target[2]}

			# Call API to move robot
			self.comms.set_position(vector_target)
			
			if input("Enter y to continue: ") != 'y':
				break	

	def get_vector_to_screw(self, box_coordinates, Zd, img2_screw_centre=None):
		# This code assumes the passing of one screws coordinates in each image img1_screw_centre[0]
		# Pixel x (Px) = (y axis distance between image center & box center * pixel size)
		# y axis label to match real world y and x

		# Add a check because this is called two different ways
		if type(box_coordinates) is tuple:
			pixel_diff_1 = (image_center[0] - box_coordinates[0], image_center[1] - box_coordinates[1])
		else:
			print("Think this line needs uncommenting or something")
			#pixel_diff_1 = (image_center[0] - box_coordinates[0][0], image_center[1] - box_coordinates[0][1])
		screw_in_camera_location = [pixel_diff_1[0] * pixel_size, pixel_diff_1[1] * pixel_size]

		camera_axis_to_robot_axis = [-1, -1]

		screw_in_camera_location = screw_in_camera_location * np.transpose(camera_axis_to_robot_axis)

		# Ratio, divide x&y by focal length, multiply by real world distance
		# Negate the pixel distance because the image is inverted so the axis are reversed
		# Also, the cameras x and y are mapped to y and x in the robot coordinates

		camera_to_screw = (screw_in_camera_location[1] * abs(Zd) / u,
							screw_in_camera_location[0] * abs(Zd) / u,
							Zd)

		return camera_to_screw
	
	# Detect objects within images and store bouning box predictions to a class variable
	def detect(self):
		# Predict bounding box objects
		self.predictions = self.model(self.image)[0]

		self.labelled_img = self.predictions.plot()

		self.num_predictions = self.predictions.boxes.data.shape[0]

		# Extract the bouding boxes from predictions and remove from gpu memory
		self.coordinates_xyxy = self.predictions.boxes.xyxy.detach().cpu().numpy()
		self.coordinates_xywhn = self.predictions.boxes.xywhn.detach().cpu().numpy()

		self.box_cls = self.predictions.boxes.cls.detach().cpu().numpy()

		self.box_conf = self.predictions.boxes.conf.detach().cpu().numpy()

	
	# Function to show the latest image labelled with the object detector
	def show_labelled_image(self, title):
		img = Image.fromarray(self.labelled_img[..., ::-1])  # RGB PIL image
		open_cv_image = np.array(img) 
		# Convert RGB to BGR 
		labelled_image = open_cv_image[:, :, ::-1].copy() 


		if title == "":
			title = "Labelled Image"
		cv2.imshow(title, labelled_image)
		if cv2.waitKey(0) == 27:
			cv2.destroyAllWindows()
			exit()

		cv2.destroyAllWindows()

	def show_image(self, title, close_window_on_key=True, terminate_on_esc=False):
		cv2.imshow(title, self.image)

		if close_window_on_key:
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		# A catch incase something incorrect was detected
		if terminate_on_esc and (input("Proceed?: ") != 'y'):
			exit()


	def find_and_move_to_screw(self):
		moving_to_screw = False
		while not moving_to_screw:
			logging.write_log("client", "\nNew Run:\n")
			try:
				# Creating an request object to store the response
				# The URL is referenced sys.argv[1]

				self.comms.get_photo()

				self.show_image(terminate_on_esc=True)
				
				# Locate and box the screws in the captured image
				self.detect()

				if self.num_predictions == 0:
					print("Error: No object was detected in one of the frames")
					continue

				## Check for detections
				self.show_labelled_image("Looking for one screw")

				# A catch incase something incorrect was detected
				if input("Proceed?: ") != 'y':
					exit()

				# Find the box center closest to the image center
				closest_box_to_center = (0, 1000, (0, 0))

				for i in range(self.num_predictions):
					# if box is of class screw, find the centre of the box
					# We are assuming only one screw is within the image and is detected correctly in both images.
					if self.predictions.cls[i] == 0:
						## Try simply finding the middle of the bounding box
						# Convert to numpy array and select the screw box (will be the first in the list if only one was detected)
						xyxy_coordinates = self.coordinates_xyxy[i]

						top_left = (xyxy_coordinates[0], xyxy_coordinates[1])
						bottom_right = (xyxy_coordinates[2], xyxy_coordinates[3])

						centre_coordinate = (top_left[0] + (bottom_right[0] - top_left[0]) / 2, top_left[1] + (bottom_right[1] - top_left[1]) / 2)
						euclidean_distance_from_center = np.linalg.norm(centre_coordinate - image_center)
						
						if euclidean_distance_from_center < closest_box_to_center[1]:
							closest_box_to_center = (i, euclidean_distance_from_center, centre_coordinate)
					#TODO: Apply some de-duplicate algorithm

				# Find closest box to centre and pass to get vector
				camera_to_screw = self.get_vector_to_screw(closest_box_to_center[2], Zd_depth)

				print("Predicted distance from camera to screw: {}".format(camera_to_screw))

				angle = self.get_wrist_angle()
				print("Wrist angle: {}".format(angle))

				if angle != None:
					# Rotate the camera to motor vector by the wrist angle
					rotated_robot_head_to_camera = self.rotate_vector(robot_head_to_camera, angle)
					print("Rotated camera to motor vector: {}".format(rotated_robot_head_to_camera))
				
				# Calculate the screw vector relative to the robot
				motor_to_screw = {'Xd': int(camera_offset[0] + (2.5*camera_to_screw[0])),
									'Yd': int(camera_offset[1] + (2.5*camera_to_screw[1])),
									'Zd': int(camera_offset[2] + camera_to_screw[2])}

				print(f"Predicted distance from robot head to screw: {motor_to_screw}")		
				
				# Cv2 display the first image with the location of the screw and the centre of the image both a dots
				cv2.circle(self.image, (int(closest_box_to_center[2][0]), int(closest_box_to_center[2][1])), 5, (0, 0, 255), -1)
				cv2.circle(self.image, (int(image_center[0]), int(image_center[1])), 5, (0, 255, 0), -1)
				
				self.show_image("Img1")

				if input("Enter y to proceed: ") == 'y':
					print("Moving to screw")
				else:
					print("Move cancelled")
					continue
				
				print(motor_to_screw)
				if self.comms.move_by_vector(motor_to_screw):
					print("Move Successful")

				cv2.destroyAllWindows()
			except Exception as e:
				print(str(e))

	# Function to rotate a vector by an angle
	def rotate_vector(self, vector, angle):
		# Convert angle to radians
		angle = math.radians(angle)

		# Create rotation matrix
		rotation_matrix = np.array([[math.cos(angle), -math.sin(angle), 0],
									[math.sin(angle), math.cos(angle), 0],
									[0, 0, 1]])

		# Multiply rotation matrix by vector
		rotated_vector = np.matmul(rotation_matrix, vector)

		return rotated_vector

	def locate_screw_rel_to_robot(self, candidate_box_data):
		# Get the robots current location
		location = self.comms.get_position()
		print("Last Successful Move: {}".format(location))

		# Calculate a vector for the screw
		camera_to_screw = self.get_vector_to_screw(candidate_box_data['centre'], Zd_depth)
		print("Predicted distance from camera to screw: {}".format(camera_to_screw))

		angle = self.get_wrist_angle()
		print("Wrist angle: {}".format(angle))

		if angle != None:
			# Rotate the camera to motor vector by the wrist angle
			rotated_robot_head_to_camera = self.rotate_vector(robot_head_to_camera, angle)
			print("Rotated camera to motor vector: {}".format(rotated_robot_head_to_camera))
		
		# - rotated_robot_head_to_camera[0]
		# Calculate the screw vector relative to the robot
		robot_to_screw = {'Xd': int(location['Xd'] + camera_offset[0] + (2.5*camera_to_screw[0])),
							'Yd': int(location['Yd'] + camera_offset[1] + (2.5*camera_to_screw[1])),
							'Zd': int(location['Zd'] + camera_offset[2] + camera_to_screw[2])}
		
		# Store the screw vector and offsets in the dictionary
		candidate_box_data['loc'] = robot_to_screw
		

	# Given an image:
	# 	Detect bounding boxes
	#	Add box details to a dictionary
	#	Identify overlapping bounding boxes
	#	Remove overlapping hole boxes and add screws to holes when overlapping.
	def search_for_screws(self):
		# Determien if two boxes overlap
		def overlap(box1, box2):
			x1, y1, x2, y2 = box1
			a1, b1, a2, b2 = box2
			a = x2<a1
			b = a2<x1
			c = y2<b1
			d = b2<y1

			the_or = not (a or b or c or d)
			return the_or # If any of these are true, they don't overlap

		
		logging_absl.write_log("client", "\nNew Run:\n")
		try:		
			# Locate and box the screws in both images
			self.detect()

			if len(self.num_predictions) != 0:
				# Structure:
				#	{index: {class: int, 'box': [x1, y1, x2, y2], 'conf': float, 'loc': [x, y, z], 'centre': [x, y]}}}
				candidate_screw_boxes = {}
				
				# Create a dictionary entry for each box prediction
				for i in range(self.num_predictions):
					# Convert to numpy array and select the screw box (will be the first in the list if only one was detected)
					xyxy_coordinates = self.coordinates_xyxy[i]
					box_cls = int(self.box_cls[i])
					conf = self.box_conf[i]

					top_left = (xyxy_coordinates[0], xyxy_coordinates[1])
					bottom_right = (xyxy_coordinates[2], xyxy_coordinates[3])

					# Calculate the center of the box
					centre = (top_left[0] + (bottom_right[0] - top_left[0]) / 2,
								top_left[1] + (bottom_right[1] - top_left[1]) / 2)

					# Add the box, we will run check later for overlap
					if len(self.bounding_boxes) != 0:
						index = max(self.bounding_boxes.keys()) + i+1
					else:
						index = i
					candidate_screw_boxes[index] = {'class': box_cls, 'box': xyxy_coordinates, 
												'conf': conf, 'loc': (0, 0, 0), 'centre': centre}
						
				
				if len(candidate_screw_boxes) == 0:
					print("No screws detected")
					return

				# Check for overlapping same class boxes
				# If the class is screw and overlaps with another screw, discard the lowest confidence screw
				# Also, if a box of class screw has a centre that is within the bounds of a box of class hole:
				# 	add the screw to the hole

				indeces_to_remove = []
				reasons_to_remove = []
				
				for candidate_box_index_1, candidate_box_data_1 in candidate_screw_boxes.items():
					for candidate_box_index_2, candidate_box_data_2 in candidate_screw_boxes.items():

						if (candidate_box_index_1 != candidate_box_index_2
							and overlap(candidate_box_data_1['box'], candidate_box_data_2['box'])
							and candidate_box_index_1 not in indeces_to_remove
							and candidate_box_index_2 not in indeces_to_remove):

							# Delete overlapping boxes of the same class, lowest confidence first
							if candidate_box_data_1['class'] == candidate_box_data_2['class'] and overlap:
								if candidate_box_data_1['conf'] > candidate_box_data_2['conf']:
									indeces_to_remove.append(candidate_box_index_2)
									reasons_to_remove.append('overlap')
								else:
									indeces_to_remove.append(candidate_box_index_1)
									reasons_to_remove.append('overlap')
							else:
								# If they are of difference classes, add the screw data as 'screw' to the box
								# 0 is screw, 1 is hole
								if candidate_box_data_1['class'] == 1:
									candidate_screw_boxes[candidate_box_index_1]['screw'] = candidate_box_data_2
									indeces_to_remove.append(candidate_box_index_2)
									reasons_to_remove.append('screw')
								else:
									candidate_screw_boxes[candidate_box_index_2]['screw'] = candidate_box_data_1
									indeces_to_remove.append(candidate_box_index_1)
									reasons_to_remove.append('screw')

				# Remove the overlapping boxes
				print(f"Removing boxes:\n")
				for index, screw in enumerate(indeces_to_remove):
					print(f"{screw}: {candidate_screw_boxes[screw]}, reason: {reasons_to_remove[index]}")
					candidate_screw_boxes.pop(screw)


				# If there are any boxes left, calculate the vector to the screw and store in the dictionary
				for i, candidate_box_data in candidate_screw_boxes.items():
					#locate_screw_rel_to_robot(candidate_box_data)
					if candidate_box_data['class'] == 1 and 'screw' in candidate_box_data.keys():
						self.locate_screw_rel_to_robot(candidate_box_data['screw'])
				
				# Display all candidate screw boxes as dots on the image
				for i, candidate_box_data in candidate_screw_boxes.items():
					# Check the boundig box is a hole and if it contains a screw
					if candidate_box_data['class'] == 1:
						if 'screw' not in candidate_box_data.keys():
							cv2.circle(self.image, (int(candidate_box_data['centre'][0]), int(candidate_box_data['centre'][1])), 5, (255, 0, 0), -1)
						if candidate_box_data['class'] == 1 and 'screw' in candidate_box_data.keys():
							# Draw a dot for the screws centre
							cv2.circle(self.image, (int(candidate_box_data['screw']['centre'][0]), int(candidate_box_data['screw']['centre'][1])), 2, (0, 0, 255), -1)
					else:
						# Draw a dot for the screws centre
						cv2.circle(self.image, (int(candidate_box_data['centre'][0]), int(candidate_box_data['centre'][1])), 2, (0, 255, 0), -1)

				self.bounding_boxes.update(candidate_screw_boxes)

			else:
				print("Error: No object was detected in one of the frames")

			cv2.destroyAllWindows()

		except Exception as e:
			print(str(e))


	def get_simple_photo(self):
		img_request = self.comms.get_simple_photo()
		logging.write_log("client", "Received Image from Server")

		if img_request.status_code == requests.codes.ok:
			# Read numpy array bytes
			np_zfile = np.load(io.BytesIO(img_request.content))

			image = np_zfile['arr_0']

			return image
		else:
			print("Error: {}".format(img_request.status_code))
			return 1


	def get_photo(self):
		return self.comms.get_photo()

	# Create a function to get the wrist angle (this is separate for simplicity)
	def get_wrist_angle(self):
		angle_request = self.comms.get_wrist_angle()
		logging.write_log("client", "Received Angle from Server")

		if angle_request.status_code == requests.codes.ok:
			# Read data
			angle = angle_request.json()['angle']

			return angle
		else:
			return None

	def try_annotate_and_save_image(self, function_args):
		(self.image, image_id) = function_args

		# Create names
		file_name = base_directory + 'images/' + str(image_id)
		image_name = file_name + ".jpg"
		annotations_name = file_name + ".txt"
		
		# Save image
		img_data = Image.fromarray(self.image)
		img_data.save(image_name)

		# If any objects are detected, save the bounding box numbers in a file associated with the image
		self.detect()

		if self.num_predictions != 0:

			# Write the annotations to the target directory in yolo format
			# Write the bounding box coordinates to a text file
			with open(annotations_name, "w") as box_file:
				for i in range(self.num_predictions):
					xywhn_coordinates = self.coordinates_xywhn[i]
					box_class = int(self.box_cls[i])

					box_file.write(str(box_class) + " " + 
										str(xywhn_coordinates[0]) + " " + 
										str(xywhn_coordinates[1]) + " " + 
										str(xywhn_coordinates[2]) + " " + 
										str(xywhn_coordinates[3]) + "\n")

	# Write a function to plan a route and move the robot to each screw	
	def plan_route_to_screws(self):
		# Create a new dictionary for the bounding boxes with index and loc
		# Structure:
		#	{index: {target_index = int, 'distance': int}}
		locations_dict = {}

		# Extract only the screw locations from the hole bounding boxes
		# Done here to allow flexibility in the bounding box structure in the previous function
		for index, box in self.bounding_boxes.items():
			if 'screw' in box.keys():
				locations_dict[index] = {'index': index, 'loc': box['screw']['loc'], 'distances': {}}

		# Add a box for the robot location
		# Request the robots current location
		location = self.comms.get_position()
		locations_dict['robot'] = {'index': 'robot', 'loc': location, 'distances': {}}

		# Calculate the euclidean distance between each screw bounding box
		for index1, box1 in locations_dict.items():
			for index2, box2 in locations_dict.items():
				if index1 != index2:
					# Calculate the euclidean distance between each box
					box1_loc = np.array([box1['loc']['Xd'], box1['loc']['Yd'], box1['loc']['Zd']])
					box2_loc = np.array([box2['loc']['Xd'], box2['loc']['Yd'], box2['loc']['Zd']])
					difference = box1_loc - box2_loc
					box1['distances'][box2['index']] = np.linalg.norm(difference)

		# Run the nearest neighbour algorithm on the points and outputting the visiting order indeces into the screw_locations array
		# Start at the box with the highest index

		# Initialise the screw_locations array and the visited bool
		screw_locations = []
		
		# Start at the robots location
		current_box = locations_dict['robot']
		screw_locations.append(locations_dict['robot']['loc'])

		# Find the nearest neighbour to the current box
		while len(screw_locations) != len(locations_dict):
			# Set the current box to visited
			current_box['visited'] = True

			# Find the nearest neighbour that hasn't been visited
			# nearest_neighbour = min(current_box['distances'], key=current_box['distances'].get)
			nearest_neighbour = min((index for index in current_box['distances'] 
							if not locations_dict[index].get('visited', False)), key=current_box['distances'].get)

			# Add the nearest neighbour to the screw_locations array
			screw_locations.append(locations_dict[nearest_neighbour]['loc'])

			# Set the current box to the nearest neighbour
			current_box = locations_dict[nearest_neighbour]

		# Print the screw locations with their indexes
		[print(f"Screw: {index}:  {location}") for index, location in enumerate(screw_locations)]

		return screw_locations

	def move_to_screws(self, screw_locations):
		try:
			# Wait till the user says go
			if input("Enter y to proceed: ") == 'y':
				print("Moving to screw")
			
			# Move to each screw in the screw_locations array
			for screw_loc in screw_locations:
				# Call API to move robot
				self.comms.set_position(screw_loc)
		except Exception as e:
			print("Move to screws failed")
			print(str(e))

	# Function to plot screw x,y coordinates on a graph representing the surface of the laptop
	def plot_screw_locations(self):
		# Iterate through self.bounding_boxes to plot the x and y of the loc of each screw
		for _, box in self.bounding_boxes.items():
			if 'screw' in box.keys():
				plt.scatter(box['screw']['loc']['Yd'], box['screw']['loc']['Xd'])

		# Show the plot
		plt.show()

	def touch_all_screws(self):
		try:
			# Create the dict to store the screw data and add it to the function args
			self.bounding_boxes = {}

			# Build list of screws
			self.roam_and_apply_function(self.search_for_screws, function_type.DETECT_MULTIPLE_SCREWS)

			# Display the x and y coordinates of the screws on a graph representing the surface of the laptop
			self.plot_screw_locations()

			# Build route to screws
			route = self.plan_route_to_screws()

			print("Route: \n{}".format(route))

			# Move to screws
			self.move_to_screws(route)
		except Exception as e:
			print("Find and contact screws failed")
			print(str(e))


	def roam_and_apply_function(self, function, function_type):

		direction = -1
		photo_gap = 50

		robot_bounds = [150, 180, 150]
		robot_location = [150, 180, 150]

		destination = {'Xd': robot_location[0], 'Yd': robot_location[1], 'Zd': robot_location[2]}
		# Call API to move robot
		self.comms.set_position(destination)

		scan_complete = False
		while not scan_complete:
			x_delta = 0
			y_delta = 0

			# Take photo
			if not self.get_photo():
				print("Failed to take photo")
				continue

			try:
				if function_type == function_type.DETECT_MULTIPLE_SCREWS:
					function_data = (function_type[1], self.image, function_type[2])
				elif function_type == function_type.GATHER_DATA:
					function_data = (function_type[1], self.image, image_id)
					image_id += 1

				function(function_data)

				# Move horizontally within Lego bounds
				if abs(robot_location[1] + (direction * photo_gap)) <= robot_bounds[1]:
					robot_location[1] += direction * photo_gap
					y_delta = direction * photo_gap

					if self.comms.move_by_vector({'Xd': x_delta, 'Yd': y_delta, 'Zd': 0}):
						print("Successful Move")
				else:  # Move up x-axis, change direction
					robot_location[0] += photo_gap
					x_delta = photo_gap
					direction = -direction

					moved_in_x = False
					while not moved_in_x:
						logging.write_log("client", "Attempt Move Robot")

						if self.comms.move_by_vector({'Xd': x_delta, 'Yd': y_delta, 'Zd': 0}):
							moved_in_x = True
						else:
							# If not passed y center, move further towards center
							if np.sign(direction) is not np.sign(robot_location[1]) and abs(
									robot_location[1] + (direction * photo_gap)) <= robot_bounds[1]:
								robot_location[1] += direction * photo_gap
								y_delta = direction * photo_gap
							# Crossed y=0 and cannot move forward in x means reached far side
							elif np.sign(direction) == np.sign(robot_location[1]) and abs(
									robot_location[1] + (direction * photo_gap)) <= robot_bounds[1]:
								scan_complete = True
								break
			except Exception as e:
				print(str(e))

		if function_type[0] == function_type.DETECT_MULTIPLE_SCREWS:
			return 
		elif function_type[0] == function_type.GATHER_DATA:
			# Call API to move robot
			self.comms.set_position(robot_default_location)

	def scan_laptops(self):
		# Prepared id for next laptop
		self.last_image_num = -1

		while True:
			another_laptop = input("Ready to scan the laptop?: ")
			if type(another_laptop) == str and (another_laptop == 'Y' or another_laptop == 'y'):
				# Check if the base directory is empty
				if os.listdir(base_directory + 'images/') == []:
					print("Scanning laptop")

					# Create a signal handler to save the yaml file
					def signal_handler(sig, frame):
						# If no yaml file exists in base directory + images, create one
						if not os.path.isfile(base_directory + 'dataset' + ".yaml"):
							print('Saving yaml file')
							# Create yaml file for the dataset
							with open(base_directory + 'dataset' + ".yaml", "w") as yaml_file:
								yaml_file.write("class_names:\n"
												"- Screw/Phillips\n"
												"- Hole\n"
												"nc: 2\n"
												"path: ..\n"
												f"test: {base_directory}Images\n")
							exit(0)

					signal.signal(signal.SIGINT, signal_handler)
				else:
					# Take the laptop id to start with as the largest file name
					self.last_image_num = max([int(file_name.split('.')[0]) for file_name in os.listdir(base_directory + 'images/')])
					self.last_image_num -= 1 # image id is incremented before saving

			else:
				print("Closing scan")
				return 0

			self.roam_and_apply_function(self.try_annotate_and_save_image, function_type.GATHER_DATA)


	def find_robot_limit(self):
		has_moved = True
		axis = 'x'
		direction = -1
		increment = 1
		y_limit = 120
		# x_limit = 150

		destination = {'Xd': 150, 'Yd': 0, 'Zd': 150}
		# Call API to move robot
		self.comms.set_position(destination)

		while has_moved:
			# @z=150 Closest x=150
			if axis == 'x':
				has_moved = self.comms.move_by_vector({'Xd': direction * 1, 'Yd': 0, 'Zd': 0})
			# @z=150 Left most y =
			elif axis == 'y':
				has_moved = self.comms.move_by_vector({'Xd': 0, 'Yd': direction * 1, 'Zd': 0})
			elif axis == 'z':
				has_moved = self.comms.move_by_vector({'Xd': 0, 'Yd': 0, 'Zd': direction * 1})
			increment += 1

			if not has_moved :
				print("Last Successful Move: {}".format(self.comms.get_position()))


	def reset_robotic_arm(self):
		# Call API to move robot
		self.comms.set_position(robot_default_location)


	def main(self, _argv):
		task = None
		while task == None:
			print("Using the standard model...\n"
				"Options, you can return to this page:\n"
				"1 - Live stream image from robot and detect screws (Unknown is this works)\n"
				"2 - Move the robot to a location\n"
				"3 - Locate a screw and move to it's location\n"
				"4 - Heuristic search laptop for screws\n"
				"5 - Scan a laptop, autodetect screws and save to dataset (Needs functionalising)\n"
				"6 - Find the robots boundaries (Only extends in x axis)\n"
				"7 - Reset the robots location\n"
				"0 - Exit\n")
			task = input("Task: ")

			if not task.isdigit():
				raise ValueError("Task should be a digit!?")
			
			task = int(task)

			if (task < 0) or (task > 7):
				print("Please enter a valid task number")
				task = None
			elif task == 1:
				print("Screws Stream")
				self.detect_screws_in_stream()
			elif task == 2:
				print("Move Robot")
				self.move_robot()
			elif task == 3:
				print("Locating a Screw")
				self.find_and_move_to_screw()
			elif task == 4:
				print("Laptop Search")
				self.touch_all_screws()
			elif task == 5:
				print("Laptop Scan")
				self.scan_laptops()
			elif task == 6:
				print("Find Robot Limit")
				self.find_robot_limit()
			elif task == 7:
				print("Resetting Robotic Arm")
				self.reset_robotic_arm()
			elif task == 0:
				print("Exiting")
				exit(0)
			task = None


if __name__ == '__main__':
	try:
		app.run(main)
	except SystemExit:
		pass
