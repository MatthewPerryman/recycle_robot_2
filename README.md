# recycle_robot_2
The second edition of the codebase for the robot system.
## Repo Aims
- Operate as a client to request data and perform calculatios to direct the server controlling the robot in a push manner.
- Enable user access to functions that control a variety of features for the robot including movement, capturing photos and locating screws.
- The functionality should support the plug and play of object detection models for the detection of objects, namely screws, within images.
- Localisation of the objects detected within the image should be calculated and used to move the robot to the screws location.

## Hardware Configuration
The codebase runs on a bespoke hardware configuration of a computer, raspberry pi and uArm robotic arm (discontinued). This repository is designed to work in tandem with the recycle_robot_server_2, where this repo is used on the computer and the server repo is deployed on the raspberry pi as a server.
The configuration looks something like this:
