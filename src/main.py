from face_detection import Face_Detection
from landmarks import LandmarksDetection
from head_pose_estimation import Head_Pose
from gaze_estimation import Gaze_Estimation
from input_feeder import InputFeeder
from openvino.inference_engine import IENetwork, IECore
import imutils
from mouse_controller import MouseController
import argparse
import cv2
import time
import json
import models_main
import matplotlib.pyplot as plt
from models_main import predict
import os
import logging

models_main.init()
logging.basicConfig(level=logging.INFO)

face_model_path="models/face_detection/face-detection-adas-0001"
landmarks_model_path="models/landmarks-regression-retail-0009/landmarks-regression-retail-0009"
hpose_model_path="models/head-pose-estimation-adas-0001/head-pose-estimation-adas-0001"
gaze_model_path="models/gaze-estimation-adas-0002/gaze-estimation-adas-0002"



def get_args():
    
    
    ap=argparse.ArgumentParser()
    #c_desc="Path to the input configuration file with model paths"
    b_desc="Option to perform benchmarking eg 16, 32,16-INT8. Use fm, hm, lm, gm to pass model paths"
    d_desc="Device type eg CPU, GPU, MYRIAD, FPGA. Default is MYRIAD"
    e_desc="Path to CPU Extension"
    l_desc="Location of input file. Leave blank for webcam"
    t_desc="type of the input_file. video, image or cam"
    p_desc="Probability threshold for face detection model"
    f_desc="Flag to display output of the intermediate models. Can be FD, FL, DP, GE"
    
    fm_desc="Path to face model for benchmarking"
    hm_desc="Path to head pose model for benchmarking"
    lm_desc="Path to landmarks detection model for benchmarking"
    gm_desc="Path to gaze detection model for benchmarking"
    
    
    ap._action_groups.pop()
    required = ap.add_argument_group('required arguments')
    optional = ap.add_argument_group('optional arguments')
    
    #required.add_argument("-c", help=c_desc)
    required.add_argument("-l", help=l_desc)
    required.add_argument("-t", help=t_desc)
    
    optional.add_argument("-b", help=b_desc, default=None)
    optional.add_argument("-d", help=d_desc, default="MYRIAD")
    optional.add_argument("-e", help=e_desc, default=None)
    optional.add_argument("-p", help=p_desc, default=0.5)
    optional.add_argument("-fm", help=fm_desc, default=face_model_path)
    optional.add_argument("-hm", help=hm_desc, default=hpose_model_path)
    optional.add_argument("-lm", help=lm_desc, default=landmarks_model_path)
    optional.add_argument("-gm", help=gm_desc, default=gaze_model_path)
    optional.add_argument("-f", help=f_desc, default=None)
    
    
    args=ap.parse_args()
    return args


def main(args):
    
    
    input_type=args.t
    input_files=args.l
    flags=args.f
    
    face_detect=Face_Detection(face_model_path, args.d, args.p, args.e)

    face_detect.load_model()

    landmarks_model=LandmarksDetection(landmarks_model_path, args.d, args.e)

    landmarks_model.load_model()

    head_pose=Head_Pose(hpose_model_path, args.d, args.e)
    head_pose.load_model()

    gaze_estimation=Gaze_Estimation(gaze_model_path, args.d, args.e)
    gaze_estimation.load_model()

    if input_type == 'cam':
        feeder = InputFeeder(input_type='cam')
    else:
        if not os.path.isfile(input_files):
            logging.error("Could not find the input file")
            exit(1)
        feed= InputFeeder(input_type='video', input_file=input_files)
    #feed=InputFeeder(input_type=input_type, input_file= input_files)

    
    try:
        feed.load_data()
    except Exception:
        logging.error("Could not load data from input file", exc_info=True)
    
    
    
    for batch in feed.next_batch():
        
        try:
            
            cropped_face, coords=face_detect.predict(batch)
            
            if type(cropped_face) == int:
                logging.info("Face not detected")
                if key == 27:
                    break
                continue
            
            cropped_left_eye, cropped_right_eye, left_eye_cord, right_eye_cord = landmarks_model.predict(cropped_face)
            head_angles = head_pose.predict(cropped_face)
            x,y = gaze_estimation.predict(cropped_left_eye, cropped_right_eye, head_angles)
        
        except Exception:
            logging.error("An error occured while running predictions", exc_info=True)
        
        if flags != 0:
            
        
            if flags == 'FD':
                cv2.rectangle(batch, (coords[0], coords[1]), (coords[2], coords[3]), (255, 0, 0), 3)
            if flags =='FL':
                cv2.rectangle(cropped_face, (left_eye_cord[0], left_eye_cord[1]), (left_eye_cord[2], left_eye_cord[3]), (255, 0, 0), 3)
                cv2.rectangle(cropped_face, (right_eye_cord[0], right_eye_cord[1]), (right_eye_cord[2], right_eye_cord[3]), (255, 0, 0), 3)
            if flags =='HP':
                cv2.putText(batch,
                "Head angles: yaw={:.2f} , pitch={:.2f}, roll={:.2f}".format(
                    head_angles[0], head_angles[1], head_angles[2]),
                            (20, 40),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1, (255, 0, 255), 2)
            if flags == 'GE':
                
                left_eye_mid_x= (left_eye_cord[2]-left_eye_cord[0])/2 + left_eye_cord[0]
                left_eye_mid_y=(left_eye_cord[3]-left_eye_cord[1])/2 + left_eye_cord[1]
                
                right_eye_mid_x=(right_eye_cord[2]-right_eye_cord[0])/2 + right_eye_cord[0]
                right_eye_mid_y=(right_eye_cord[3]- right_eye_cord[1])/2 + right_eye_cord[1]
                
                left_eye_new_x=int(left_eye_mid_x + x*160)
                left_eye_new_y=int(left_eye_mid_y + y*160*-1)
                right_eye_new_x=int(right_eye_mid_x + x*160)
                right_eye_new_y=int(right_eye_mid_y + y*160*-1)
                cv2.line(cropped_face, (int(left_eye_mid_x), int(left_eye_mid_y)), (int(left_eye_new_x), int(left_eye_new_y)), (255, 0, 255), 5)
                cv2.line(cropped_face, (int(right_eye_mid_x), int(right_eye_mid_y)), (int(right_eye_new_x), int(right_eye_new_y)), (255, 0, 255), 5)
                
        

                
                
        mouse=MouseController(precision='low', speed='fast')
        mouse.move(x,y)    
        
        
        batch = imutils.resize(batch, width=500)
        cv2.imshow('frame', batch)
        key = cv2.waitKey(1) & 0xFF
    feed.close()
        
def benchmark(args):
    print("runing benchmark")
    #file=open(args.c)
    #confs=json.loads(file.read())
    

    
    input_type=args.t
    input_files=args.l
    
    
    face_lt_start=time.time()
    face_detect=face_detection(args.fm, args.d, args.p, args.e)

    face_detect.load_model()
    face_lt=time.time()-face_lt_start
    
    
    landmark_lt_start=time.time()
    landmarks_model=LandmarksDetection(args.lm, args.d, args.e)

    landmarks_model.load_model()
    landmark_lt=time.time()-landmark_lt_start

    
    head_pose_lt_start=time.time()
    head_pose=Head_Pose(args.hm, args.d, args.e)
    head_pose.load_model()
    head_pose_lt=time.time()-head_pose_lt_start

    
    gaze_lt_start=time.time()
    gaze_estimation=Gaze_Estimation(args.gm, args.d, args.e)
    gaze_estimation.load_model()
    gaze_lt=time.time()-gaze_lt_start


    feed=InputFeeder(input_type='video', input_file=input_files)

    feed.load_data()

    for batch in feed.next_batch():
        
        face_inf_start=time.time()
        cropped_face=face_detect.predict(batch)
        face_inf_time=time.time()-face_inf_start
      
        landmark_inf_start=time.time()
        cropped_left_eye, cropped_right_eye = landmarks_model.predict(cropped_face)
        landmark_inf_time=time.time()-landmark_inf_start
        
        
        head_pose_inf_start=time.time()
        head_angles = head_pose.predict(cropped_face)
        head_pose_inf_time=time.time()-head_pose_inf_start
        
        
        gaze_inf_start=time.time()
        x,y = gaze_estimation.predict(cropped_left_eye, cropped_right_eye, head_angles)
        gaze_inf_time=time.time()-gaze_inf_start
        
        
        #plotting load_time
        models=['Face_detect', 'landmark_detect', 'Head_pose_est', 'Gaze est']
        loading_times=[face_lt, landmark_lt, head_pose_lt, gaze_lt]
        plot_loading_time(models, loading_times, args.b)
        
        #plotting inference_time
        inference_times=[face_inf_time, landmark_inf_time, head_pose_inf_time, gaze_inf_time]
        plot_inf_time(models, inference_times, args.b)
        
        logging.info("Benchmarking done!")
        
        

        break
    feed.close()
        
        
    
def plot_inf_time(x, y, b):
    plt.bar(x, y)
    plt.ylabel("Inference times for FP:"+b)
    plt.savefig('benchmarks/'+b+'/inference.jpg')
    plt.close()
    
def plot_loading_time(x, y, b):
    plt.bar(x, y)
    plt.ylabel("Loading times for FP"+b)
    plt.savefig('benchmarks/'+b+'/loading.jpg')
    plt.close()

    
    


if __name__== '__main__':
    args=get_args()
   
    
    if args.b:
        benchmark(args)
    else:
        main(args)








