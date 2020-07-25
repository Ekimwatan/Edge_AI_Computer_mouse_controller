# Computer Pointer Controller

## Description

This project uses a combination of 4 models and a raspberry pi 4b to eventually control the mouse on your screen.
The project will include building of a model inference pipeline which takes in input from either a webcam or a video file,
extracts features such a face location, eye location, head pose and eventually gaze estimation. This is then
used to direct the mouse to the location the person is gazing at.
The project makes use of pre-trained models from the intel openvino model zoo.

These models are:
-face-detection-adas-0001
-landmarks-regression-retail-0009
-head-pose-estimation-adas-0001
-gaze-estimation-adas-0002

## Project Set Up and Installation

### Prerequisites

- Intel OpenVINO Toolkit for raspberry pi or PC. 
You can get the installation instructions on the link below

Linux: 
https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html

Windows:
https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_windows.html
Raspbery pi:https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_raspbian.html

- Python 3.7



### Setup

- Set up a virtual environment eg using below code for conda:

`conda create -n yourenvname python=3.7 anaconda`

- Activate virtual environemt using below code

`source activate yourenvname`

- Download or clone the repository

- Source OpenVino Environment

This is done by entering the following to your cmd `source /opt/intel/openvino/bin/setupvars.sh`

- Installing the requirements in requirements.txt file

- Download the models from the intel moddelzoo from the link below

https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/


## Demo
To run a demo, 
- navigate to this directory ../mouse_pointer/starter/src
- Run 

`python python main.py -t video -l /home/pi/mouse_countroller/mouse_pointer/starter/bin/demo.mp4 -d CPU -b 16`



## Documentation

This project allows a number of command line arguments to achieve the follwoing:

- Select an input type ie Image, Video or cam `-t`. 
- Pass input file location for video or image `-l`
- Select the device to use. This can be CPU, FPGA, GPU, or MYRIAD `-d`
- Option to perform a benchmark. This takes the precision of the models eg FP(16, 32,16-INT8) `-b`
- Pass CPU extension path for custom models `-e`
- Probability threshold for face detection model `-p`
- Path to face detection model for benchmarking `-fm`
- Path to Facial landmarks detection model `-lm`
- Path to Head pose estimation model `-hm`
- Path to Gaze estimation model  `-gm`

The directory structure is as the below image:

 ![Project Directory](./mouse_pointer/starter/src/directory.png)


## Directory Overview

- Benchmarks: Contains the outputs/graphs from benchmarking the different model precisions
- Face_detection.py: Contains code for the face detection model
- gaze_detection.py: Contains code for the gaze estimation model
- head_pose_estimation: contains code for the head pose estimation model
- landmarks.py: contains code for the facial landmarks detection model
- main.py: Contains code for the application
- models_main: Contains code for initiating the openvino IECore
- Mouse_controller: Contains code that controls the mouse on the screen


## Benchmarks

The benchmarking results for the FP16 model that is supported by the raspberry pi are as follows:
The face detection model had the lowest loading time as compared to the other models.
The face detection model had the highest inference time as compared to the other models
which could be slowing down the application.

The detailed graphs can be found below and in /benchmarks/16/

![](./benchmarks/16/inference.jpg)
![](./benchmarks/16/loading.jpg)




### Edge Cases
The application would struggle in an instance where there are more than one person in the frame. This is because
it can only handle one face at a time. So only one case will be used.