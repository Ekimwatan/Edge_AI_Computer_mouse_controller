
from openvino.inference_engine import IENetwork, IECore
import models_main
import cv2
import logging
logging.basicConfig(level=logging.INFO)


class Head_Pose:
  
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
            logging.info("Unsupported layers were found. They are {}".format(unsupported_layers))


        
        try:
            self.net=models_main.core.load_network(self.model, self.device, num_requests=1)
        except Exception:
            logging.error("Could not load network..", exc_info=True)
        
        logging.info("Head pose model loaded...")
        self.input_name=next(iter(self.net.inputs))
        self.output_name=next(iter(self.net.outputs))
        self.input_shape=self.net.inputs[self.input_name].shape
        self.output_shape=self.net.outputs[self.output_name].shape
        


    def predict(self, image):
        
        #preprocess cropped face
        (n, c, h, w) = self.net.inputs[self.input_name].shape
        
        
        frame=cv2.resize(image, (w, h))
        frame=frame.transpose((2, 0, 1))
        frame=frame.reshape(1, *frame.shape)
        
        
        input_dict={self.input_name:frame}
        try:
            output=self.net.infer(input_dict)
        except Exception:
            raise ValueError("Could not run inference..", exc_info=True)
        
     

        output=self.net.infer(input_dict)
        yaw=output["angle_y_fc"][0][0]
        pitch=output["angle_p_fc"][0][0]
        roll=output["angle_r_fc"][0][0]
    
        head_angles=[yaw, pitch, roll]
        
        
        return head_angles
        

    def check_model(self):
        pass
        

    def preprocess_input(self, image):
    
        p_image=cv2.resize(image, self.input_shape[3], self.input_shape[4])
        
        p_image=p_image.transpose((2, 0, 1))
        p_image=p_image.reshape(1, *p_image.shape)
        
        return p_image

    def preprocess_output(self, outputs):
    
        pass
