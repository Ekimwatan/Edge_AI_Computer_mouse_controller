from openvino.inference_engine import IENetwork, IECore
import models_main
import cv2
import logging
logging.basicConfig(level=logging.INFO)



class Gaze_Estimation:
   
    def __init__(self, model_path, device, extensions):
        
        self.model_path= model_path
        self.net=None
        
        self.input_name=None
        self.output_name=None
        self.exec_network=None
        self.input_shape=None
        self.output_shape=None
        self.device=device
        self.extensions=extensions
      
        

    def load_model(self):
        
        model_xml=self.model_path + ".xml"
        model_bin=self.model_path + ".bin"
        
        self.model=IENetwork(model=model_xml, weights=model_bin)
        
        ##handle extensions
        if self.extensions and self.device=="CPU":
            models_main.core.add_extension(extension_path=self.extensions, device=self.device)

        # Unsupported layers
        layers_list=models_main.core.query_network(network=self.model, device_name=self.device)
        unsupported_layers=[l for l in self.model.layers.keys() if l not in layers_list]
        if len(unsupported_layers)>0:
            logging.info("Unsupported layers were found. Check your extensions path. They are {}".format(unsupported_layers))


        
        try:
            self.net=models_main.core.load_network(self.model, self.device, num_requests=1)
        except Exception:
            logging.error("Could not load network check..", exc_info=True)
        
        logging.info("Gaze Estimation model loaded...")
        
        self.input_name=next(iter(self.net.inputs))
        self.output_name=next(iter(self.net.outputs))
        self.input_shape=self.net.inputs[self.input_name].shape
        self.output_shape=self.net.outputs[self.output_name].shape        
        

    def predict(self, cropped_left_eye, cropped_right_eye, head_angles):
        
        
        (n, c, h, w) = self.net.inputs["left_eye_image"].shape
        (n1, c1, h1, w1) = self.net.inputs["right_eye_image"].shape
        #preprocess cropped left eye
        p_left_eye=cv2.resize(cropped_left_eye, (w, h))
        p_left_eye=p_left_eye.transpose((2, 0, 1))
        p_left_eye=p_left_eye.reshape(1, *p_left_eye.shape)
        
        #preprocessing cropped right eye
        p_right_eye=cv2.resize(cropped_right_eye, (w1, h1))
        p_right_eye=p_right_eye.transpose((2, 0, 1))
        p_right_eye=p_right_eye.reshape(1, *p_right_eye.shape)
        
        #input dictionary
        
        
        input_dict={"left_eye_image":p_left_eye, "right_eye_image":p_right_eye, "head_pose_angles":head_angles}
        
        gaze_output=self.net.infer(input_dict)
        x=round(gaze_output["gaze_vector"][0][0], 4)
        y=round(gaze_output["gaze_vector"][0][1], 4)
        
        return x, y
        
        

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
    
        raise NotImplementedError

    def preprocess_output(self, outputs):
   
        raise NotImplementedError
