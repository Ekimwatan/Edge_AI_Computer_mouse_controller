
from openvino.inference_engine import IENetwork, IECore





def init():
    global core
    core = IECore()


def predict(batch):
    
    cropped_face=Face_Detect.predict(batch)
    cropped_left_eye, cropped_right_eye = landmarks_model.predict(cropped_face)
    head_angles = head_pose.predict(cropped_face)
    x,y = gaze_estimation.predict(cropped_left_eye, cropped_right_eye, head_angles)
    
    return cropped_face, cropped_left_eye, cropped_right_eye, head_angles, x, y


    
    
    
    
    