from openvino.inference_engine import IENetwork, IECore
import cv2
import imutils
from imutils.video import FPS
import time
from mouse_controller import MouseController


model_xml='models/face_detection/face-detection-adas-0001.xml'
model_bin='models/face_detection/face-detection-adas-0001.bin'
device='MYRIAD'
video_path='/home/pi/mouse_countroller/starter/bin/demo.mp4'



landmark_xml='models/landmarks-regression-retail-0009/landmarks-regression-retail-0009.xml'
landmark_bin='models/landmarks-regression-retail-0009/landmarks-regression-retail-0009.bin'

head_pose_xml='models/head-pose-estimation-adas-0001/head-pose-estimation-adas-0001.xml'
head_pose_bin='models/head-pose-estimation-adas-0001/head-pose-estimation-adas-0001.bin'


gaze_est_xml='models/gaze-estimation-adas-0002/gaze-estimation-adas-0002.xml'
gaze_est_bin='models/gaze-estimation-adas-0002/gaze-estimation-adas-0002.bin'

print("INFO: Starting to load models.....")

core=IECore()
model=IENetwork(model_xml, model_bin)
net=core.load_network(model, device, num_requests=1 )
fps = FPS().start()

layers_list=core.query_network(network=model, device_name=device)
print(layers_list)

model2=IENetwork(landmark_xml, landmark_bin)
net2=core.load_network(model2, device, num_requests=1)


model3=IENetwork(head_pose_xml, head_pose_bin)
net3=core.load_network(model3, device, num_requests=1)

model4=IENetwork(gaze_est_xml, gaze_est_bin)
net4=core.load_network(model4, device, num_requests=1)

print("INFO: All models loaded...")


input_name=next(iter(net.inputs))
input_shape=net.inputs[input_name].shape
output_name=next(iter(net.outputs))

input_name2=next(iter(net2.inputs))
input_shape2=net2.inputs[input_name2].shape
output_name2=next(iter(net2.outputs))
output_shape2=net2.outputs[output_name2].shape


input_name3=next(iter(net3.inputs))
input_shape3=net3.inputs[input_name3].shape
output_name3=next(iter(net3.outputs))
output_shape3=net3.outputs[output_name3].shape


input_name4=next(iter(net4.inputs))
input_shape4=net4.inputs[input_name4].shape
output_name4=next(iter(net4.outputs))
output_shape4=net4.outputs[output_name4].shape






(n, c, h, w) = net.inputs[input_name].shape
(n2, c2, h2, w2) = net2.inputs[input_name2].shape
(n3, c3, h3, w3) = net3.inputs[input_name3].shape
(n4, c4, h4, w4) = net4.inputs["left_eye_image"].shape
(n5, c5, h5, w5) = net4.inputs["right_eye_image"].shape


print("INFO: Opening video....")
cap=cv2.VideoCapture(video_path)

cap.open(video_path)

while cap.isOpened():
    
    flag, frame=cap.read()
    if not flag:
        break
    
    #Face detection model
    (height, width) = frame.shape[:-1]
    x_frame = imutils.resize(frame, width=500)
    p_frame=cv2.resize(frame, (w, h))
    p_frame=p_frame.transpose((2, 0, 1))
    p_frame=p_frame.reshape(1, *p_frame.shape)
    input_dict={input_name:p_frame}
    
    #face detection inference
    output=net.infer(input_dict)
    
    
    #print(output[output_name][0][0])
    coords=[]
    for box in output[output_name][0][0]:
        if box[2] > 0.50:
            xmin=box[3]*width
            ymin=box[4]*height
            xmax=box[5]*width
            ymax=box[6]*height
            coords.append((int(xmin), int(ymin), int(xmax), int(ymax)))
            
    #print(coords[0][3])
    coords=coords[0]
    cropped_face=frame[coords[1]:coords[3], coords[0]:coords[2]]
    print("cropped face returned")
    #cv2.imshow("face", cropped_face)
    
    #Landmark detection model
    (height2, width2) = cropped_face.shape[:-1]
    scnd_frame=cv2.resize(cropped_face, (w2, h2))
    scnd_frame=scnd_frame.transpose((2, 0, 1))
    scnd_frame=scnd_frame.reshape(1, *scnd_frame.shape)
    input_dict2={input_name2:scnd_frame}
    
    output2=net2.infer(input_dict2)
    result2=output2[output_name2]
    
    
    #Head pose estimation model
    (height3, width3) = cropped_face.shape[:-1]
    thrd_frame=cv2.resize(cropped_face, (w3, h3))
    thrd_frame=thrd_frame.transpose((2, 0, 1))
    thrd_frame=thrd_frame.reshape(1, *thrd_frame.shape)
    input_dict3={input_name3:thrd_frame}
    
    output3=net3.infer(input_dict3)
    yaw=output3["angle_y_fc"][0][0]
    pitch=output3["angle_p_fc"][0][0]
    roll=output3["angle_r_fc"][0][0]
    
    head_angles=[yaw, pitch, roll]
    

    #print(output_shape2)
    #print("Second output {}".format(output2[output_name2][0][1][0][0]*200))
    left_eye=[result2[0][0][0][0]*width2, result2[0][1][0][0]*height2]
    right_eye=[result2[0][2][0][0]*width2, result2[0][3][0][0]*height2]
    #print(right_eye[0])
    #print("left_eye : {} right eye: {}".format(left_eye, right_eye))
    
    
    #cropping left eye
    x_left_eye=left_eye[0]
    y_left_eye=left_eye[1]
    cropped_left_eye=cropped_face[int(y_left_eye-15):int(y_left_eye+15), int(x_left_eye-15):int(x_left_eye+15)]
    
    
    #cropping right eye
    x_right_eye=right_eye[0]
    y_right_eye=right_eye[1]
    cropped_right_eye=cropped_face[int(y_right_eye-15):int(y_right_eye+15), int(x_right_eye-15):int(x_right_eye+15)]
    #cv2.imshow("right_eye", cropped_right_eye)
    
    
    #Gaze Estimation
    #preprocessing cropped_left_eye
    #(height3, width3) = cropped_face.shape[:-1]
    
    p_left_eye=cv2.resize(cropped_left_eye, (w4, h4))
    p_left_eye=p_left_eye.transpose((2, 0, 1))
    p_left_eye=p_left_eye.reshape(1, *p_left_eye.shape)
    
    
    #preprocessing cropped_right_eye
    p_right_eye=cv2.resize(cropped_right_eye, (w5, h5))
    p_right_eye=p_right_eye.transpose((2, 0, 1))
    p_right_eye=p_right_eye.reshape(1, *p_right_eye.shape)
    
    
    dict4={"left_eye_image":p_left_eye, "right_eye_image":p_right_eye, "head_pose_angles":head_angles}
    
    
    gaze_output=net4.infer({"left_eye_image":p_left_eye, "right_eye_image":p_right_eye, "head_pose_angles":head_angles})
    x=round(gaze_output["gaze_vector"][0][0], 4)
    y=round(gaze_output["gaze_vector"][0][1], 4)
    
    
    #Controlling the mosue
    #mouse=MouseController(precision='low', speed='fast')
    #mouse.move(x,y)
    
    #cv2.imshow("Frame", frame)
    
    
    
    key = cv2.waitKey(1) & 0xFF
    #print(coords)
    fps.update()

print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))          
    
    








