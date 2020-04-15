rm -rf models/detection
rm -rf models/features
rm -rf models/estimation

mkdir -p models/detection
cd models/detection

wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/tf_text_graph_common.py
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/tf_text_graph_ssd.py

wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar xvzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
mv ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb ssd_mobilenet_v2_coco_2018_03_29.pb
wget https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt
rm ssd_mobilenet_v2_coco_2018_03_29.tar.gz
rm -rf ssd_mobilenet_v2_coco_2018_03_29/

git clone https://github.com/victordibia/handtracking.git
mv handtracking/hand_inference_graph/frozen_inference_graph.pb hand_detection.pb

wget https://raw.githubusercontent.com/victordibia/handtracking/master/model-checkpoint/ssdlitemobilenetv2/pipeline.config
mv pipeline.config hand_detection_pipeline.config

python tf_text_graph_ssd.py \
	--input hand_detection.pb \
	--output hand_detection.pbtxt \
	--config hand_detection_pipeline.config

rm -rf handtracking/

wget https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/opencv_face_detector.pbtxt
wget https://github.com/opencv/opencv_3rdparty/raw/8033c2bc31b3256f0d461c919ecc01c2428ca03b/opencv_face_detector_uint8.pb

wget https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt
wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel

cd -
mkdir -p models/estimation
cd models/estimation
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
rm shape_predictor_68_face_landmarks.dat.bz2

cd -
mkdir models/features
cd models/features
wget https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7
