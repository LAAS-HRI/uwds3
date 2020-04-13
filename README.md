
# Underworlds
A framework for physical, spatial and semantic reasoning for Human-Robot interactions.

This software is composed by two data-structures:
  1. A scene graph composed by scene nodes that contains the geometric information.
  2. A timeline of temporal situations that contains temporal predicates and captions.

It also contains differents modules to generate and maintain the scene graph and the timeline from the robot camera image by integrating a state-of-the-art CPU based perception pipeline alongwith a real-time physics engine and a tensor based probabilistic knowledge base.

Main features:
 * Lightweight CPU perception pipeline that use SSD detectors with kalman+medianflow trackers to let the GPU for the physics engine.
 * Give to the robot physical sense of his body by using the simulation engine at runtime.
 * Compute the view of the scene graph from any pose in the 3d space to latter process with image-based reasoning.
 * Detect, explain and repairs beliefs divergeance by monitoring the view of the human it interact with (called the perspective).
 * Correct small inconsistencies and infer the position of objects beyond the camera FOV by using physical reasoning.
 * Reason about occlusion, motion and geometry to detect tabletop actions.


More information in the [documentation](#).

# Quick start
**Note:** We assume that you have ROS installed, otherwise install it by following the instructions [here](https://wiki.ros.org/ROS/Installation).

Follow this instructions to quickly install :
```shell
cd ~/catkin_ws/src # go to your catkin workspace src folder
git clone https://github.com/LAAS-HRI/uwds3_msgs.git # clone the msgs definitions
git clone https://github.com/LAAS-HRI/uwds3.git # clone the lib repo
cd uwds3 # go to the lib repo root
./download_models.sh # download dnn models
./install_dependencies.sh # install dependencies
cd ..
catkin build # build the project
```

# Quick launch

In different shells, run the following:
```shell
roslaunch uwds3 r2d2_upload.launch # upload a fake R2D2 robot
roslaunch uwds3 camera_publisher.launch # use the laptop/usb camera to fake the robot sensor
roslaunch uwds3 dialogue_pipeline.launch # launch the demo pipeline
```

Open Rviz in another shell with the command `rviz`, and then add three display:
* Image display (topic : `tracks_image`)
* Image display (topic : `person_view`)
* MarkerArray display (topic : `tracks_markers`)

### Any problem ?

Please fill an issue with the error log and an example of code to reproduce the bug.

### How to contribute

To contribute to the project, fork the repo and make a pull request

## References
