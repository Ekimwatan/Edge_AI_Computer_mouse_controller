from openvino.inference_engine import IENetwork, IECore
import models_main
import cv2
import logging

logging.basicConfig(level=logging.INFO)

class LandmarksDetection:
  
  
    def __init__(self, model_path, device, extensions):
        
        self.net=None
        self.model=None
      
        self.input_name=None
        self.output_name=None
        self.input_shape=None
        self.output_shape=None
        self.device=device
        self.model_path=model_path
        self.extensions=extensions

    def load_model(self):
        
        
        model_xml= self.model_path + ".xml"
        model_bin= self.model_path + ".bin"
        self.model=IENetwork(model=model_xml, weights=model_bin)
        logging.info("Landmark Detection model initiated....")

        ##handle extensions
        if self.extensions and self.device=="CPU":
            models_main.core.add_extension(extension_path=self.extensions, device=self.device)

        # Unsupported layers
        layers_list=models_main.core.query_network(network=self.model, device_name=self.device)
        unsupported_layers=[l for l in self.model.layers.keys() if l not in layers_list]
        if len(unsupported_layers)>0:
            logging.info("Unsupported layers were found. They are {}".format(unsupported_layers))

        
        
        
        try:
            self.net=models_main.core.load_network(self.model, self.device, num_requests=1)
        
        except Exception:
            
            logging.error("could not load model. Check{}", exc_info=True)
        else:
            logging.info("Landmark detection model loaded ....")
             
        self.input_name=next(iter(self.net.inputs))
        self.output_name=next(iter(self.net.outputs))
        self.input_shape=self.net.inputs[self.input_name].shape
        self.output_shape=self.net.outputs[self.output_name].shape   

    def predict(self, image):
        #preprocess the cropped face
        
        (n, c, h, w) = self.net.inputs[self.input_name].shape
        frame=cv2.resize(image, (w, h))
        frame=frame.transpose((2, 0, 1))
        frame=frame.reshape(1, *frame.shape)
        
        
        
        
        input_dict={self.input_name:frame}
        try:
            output=self.net.infer(input_dict)
        except Exception:
            logging.error("Could not run inference..", exc_info=True)
        
        
        (height, width) = image.shape[:-1]
        results=output[self.output_name]
        left_eye=[results[0][0][0][0]*width, results[0][1][0][0]*height]
        right_eye=[results[0][2][0][0]*width, results[0][3][0][0]*height]

        #cropping left eye
        x_left_eye=left_eye[0]
        y_left_eye=left_eye[1]
        cropped_left_eye=image[int(y_left_eye-15):int(y_left_eye+15),
                                      int(x_left_eye-15):int(x_left_eye+15)]
    

        #cropping right eye
        x_right_eye=right_eye[0]
        y_right_eye=right_eye[1]
        cropped_right_eye=image[int(y_right_eye-15):int(y_right_eye+15),
                                       int(x_right_eye-15):int(x_right_eye+15)]
        
        left_eye_cord=[int(x_left_eye-15), int(y_left_eye-15), int(x_left_eye+15), int(y_left_eye+15)]
        right_eye_cord=[int(x_right_eye-15), int(y_right_eye-15), int(x_right_eye+15), int(y_right_eye+15)]
        
        


        return cropped_left_eye, cropped_right_eye, left_eye_cord, right_eye_cord

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
    
        (n, c, h, w) = self.net.inputs[self.input_name].shape
        (height, width) = image.shape[:-1]
        #p_frame = imutils.resize(frame, width=500)
        p_frame=cv2.resize(image, (w, h))
        p_frame=p_frame.transpose((2, 0, 1))
        p_frame=p_frame.reshape(1, *p_frame.shape)

    def preprocess_output(self, outputs):
        pass
        
        

        
    
     
