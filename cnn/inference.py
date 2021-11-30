import tensorflow as tf
import numpy as np
import superpixel
from skimage.measure import regionprops
from PIL import Image
import os
from glob import glob
import cv2
from config import cfg
import torch
import math


model = torch.hub.load('ultralytics/yolov5','custom', path='/home/ubuntu/dev/cordyceps/cnn/last.pt', force_reload=True)
n_segments = 25


class Inference():
    
    def use_gpu(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in tf.config.experimental.list_physical_devices("GPU"):
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print("GPU used!")
            except RuntimeError as e:
                print(e)
    
    
    def preprocess_image(self,file):
        sp = superpixel.Superpixel()
        results = model(file)
        object = results.crop(save=False)
        image = list()
        for obj in object:
            image.append((obj['im'], sp.makeslic(obj['im'],n_segments)))
        
        return image

    
    def preprocess_segment(self,image,props,idx,size):
        
        # crop segment
        cy, cx = props.centroid
        left = cx - int(size / 2)
        right = left + size
        top = cy - int(size / 2)
        bottom = top + size
        props_image = Image.fromarray(image)
        cropped_img = props_image.crop((left, top, right, bottom))
        
        # resize image
        cropped_img = np.array(cropped_img)
        cropped_img = Image.fromarray(cropped_img)
        resized_img = cropped_img.resize((227,227))
        resized_img = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2HSV)
        
        # save segments to check
        segment_image_file = os.path.join(cfg['base']['path'],cfg['inference']['dir']['seg'],'{:06d}.jpg'.format(idx))
        cv2.imwrite(segment_image_file,resized_img)
        
        resized_img = np.expand_dims(resized_img, axis=0).astype(np.float32) / 255.0
        
        return resized_img
        

    def inference(self,image_name,image,slic):
        image=np.array(image)
        class_name = ['contaminated','uncontaminated']
        regions = regionprops(slic,intensity_image=image)
        contaminated=[]
        uncontaminated=[]
        size=len(slic)//math.sqrt(n_segments)
        
        Model = tf.keras.models.load_model(os.path.join(cfg['base']['path'],cfg['model']['dir'],cfg['model']['version']))
        for idx,props in enumerate(regions):
            input_img = self.preprocess_segment(image,props,idx,size)
            prediction = Model.predict(input_img)
            
            if class_name[np.argmax(prediction)]=='contaminated':
                contaminated.append(idx)
            else:
                uncontaminated.append(idx)

        print(image_name.split('/')[-1])
        if len(contaminated) > len(uncontaminated):
            print('result: contaminated')
        else:
            print('result: uncontaminated')
        print("contaminated:",contaminated)
        print("uncontaminated:",uncontaminated)
        print()
    
    
    def process(self):
        glob_path = os.path.join(cfg['base']['path'],cfg['inference']['dir']['raw'],'*')
        image_datas = glob(glob_path)

        for image_name in image_datas:
            image = self.preprocess_image(image_name)
            for img,slic in image:
                self.inference(image_name,img,slic)
    
    
    def __init__(self):
        self.use_gpu()
        self.process()
        

if __name__ == '__main__':
    Inference()
