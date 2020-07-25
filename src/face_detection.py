
from openvino.inference_engine import IENetwork, IECore
import models_main
import cv2

import logging
logging.basicConfig(level=logging.INFO)


class Face_Detection:
   
    def __init__(self, model_path, device, threshold, extensions):
      
        
        self.net=None
        self.model=None
       
        self.input_name=None
        self.output_name=None
        self.input_shape=None
        self.output_shape=None
        self.device=device
        self.model_path=model_path
        self.threshold=threshold
        self.extensions=extensions
        
        

    def load_model(self):
      
        model_xml= self.model_path + ".xml"
        model_bin= self.model_path + ".bin"
        
        
        self.model=IENetwork(model=model_xml, weights=model_bin)
        logging.info("Face Detection Network initialized...")
        
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
        else:
            
            logging.info("Face detection Network Loaded...")
        self.input_name=next(iter(self.net.inputs))
        self.output_name=next(iter(self.net.outputs))
        self.input_shape=self.net.inputs[self.input_name].shape
        self.output_shape=self.net.outputs[self.output_name].shape

    def predict(self, image):
        
        (height, width) = image.shape[:-1]
        p_image=self.preprocess_input(image)
        input_dict={self.input_name:p_image}
        try:
            outputs=self.net.infer(input_dict)
        except Exception as e:
            raise ValueError("Could not run inference")
        
        coords=[]
        for box in outputs[self.output_name][0][0]:
            if box[2] > self.threshold:
                xmin=box[3]*width
                ymin=box[4]*height
                xmax=box[5]*width
                ymax=box[6]*height
                coords.append((int(xmin), int(ymin), int(xmax), int(ymax)))
                
        #print(coords[0][3])
        coords=coords[0]
        
        cropped_face=image[coords[1]:coords[3], coords[0]:coords[2]]
        return cropped_face, coords
        
        


        

    def preprocess_input(self, image):
   
        (n, c, h, w) = self.net.inputs[self.input_name].shape
        
        #p_frame = imutils.resize(frame, width=500)
        p_frame=cv2.resize(image, (w, h))
        p_frame=p_frame.transpose((2, 0, 1))
        p_frame=p_frame.reshape(1, *p_frame.shape)
        
        return p_frame
            
        

    def preprocess_output(self, outputs):
        
   
        coords=[]
        for box in outputs[self.output_name][0][0]:
            if box[2] > 0.50:
                xmin=box[3]*width
                ymin=box[4]*height
                xmax=box[5]*width
                ymax=box[6]*height
                coords.append((int(xmin), int(ymin), int(xmax), int(ymax)))
                
        #print(coords[0][3])
        coords=coords[0]
        cropped_face=frame[coords[1]:coords[3], coords[0]:coords[2]]
        return cropped_face
            

