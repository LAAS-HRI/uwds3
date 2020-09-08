
# Underworlds
A framework for physical, spatial and semantic reasoning for Human-Robot interactions.

This software is composed by two data-structures:
  1. A scene graph composed by scene nodes that contains the geometric and visual information.
  2. A timeline of temporal situations that contains temporal events that can represent an action, a statement or a caption.

It also contains differents modules to generate and maintain the scene graph and the timeline from the robot camera image by integrating a CPU based perception pipeline alongwith a real-time physics engine and a probabilistic triplet store.

Main features:
 - [x] Lightweight CPU perception pipeline that use SSD detectors with kalman+medianflow trackers to let the GPU for the physics engine.
 - [x] Give to the robot physical sense of his body by using the simulation engine at runtime and the `/joint_states` topic published by the robot.
 * [x] Compute the view of the scene graph from any pose in the 3d space to latter process with image-based reasoning.
 * [x] Detect, explain and repairs beliefs divergeance by monitoring the view of the human it interact with (called the perspective).
 * [x] Correct small inconsistencies and infer the position of objects beyond the camera field of view by using physical reasoning and detect tabletop actions by analyzing inconsistency.
 * [ ] [WIP] Ground verbal expressions with rare/unknown or incomplete words by using fasttext embedding in order to match with current situations or nodes

More information in the [documentation](https://github.com/LAAS-HRI/uwds3/wiki).

# Quick start
**Note:** We assume that you have ROS installed, otherwise install it by following the instructions [here](https://wiki.ros.org/ROS/Installation).

Follow this instructions to quickly install :
```shell
cd ~/catkin_ws/src # go to your catkin workspace src folder
git clone https://github.com/LAAS-HRI/uwds3_msgs.git # clone the msgs definitions
git clone https://github.com/LAAS-HRI/uwds3.git # clone the lib repo
cd uwds3 # go to the lib repo root
./download_models.sh # download dnn models
./download_word_embeddings.sh #(optional) If you want to use static word embeddings
./install_dependencies.sh # install dependencies
cd ..
catkin build # build the project to install the pyuwds3 lib
```


# Quick launch

In different shells, run the following:
```shell
roslaunch uwds3 r2d2_upload.launch # upload a fake R2D2 robot
roslaunch uwds3 camera_publisher.launch # use the laptop/usb camera to fake the robot sensor
roslaunch uwds3 dialogue_pipeline.launch # launch the demo pipeline
```

**IMPORTANT**: To be able to load the robot meshes in bullet you need to have the description package in the `catkin_ws/src` folder (exept for r2d2 that is composed only by simple primitives).

Open Rviz in another shell with the command `rviz`, and then add three display:
* Image display (topic : `myself_view`)
* Image display (topic : `other_view`)
* MarkerArray display (topic : `tracks_markers`)

you can monitor the output topic of type `WorldStamped` by running:
```shell
rostopic echo /tracks
```

### Any problem ?

Please fill an issue with the error log and an example of code to reproduce the bug.

### How to contribute

To contribute to the project, fork the repo and make a pull request
