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


YOLO = torch.hub.load('ultralytics/yolov5','custom', path=os.path.join(cfg['base']['path'],'last.pt'), force_reload=True)
TENSORFLOW = tf.keras.models.load_model(os.path.join(cfg['base']['path'],cfg['model']['dir'],cfg['model']['version']))
N_SEGMENTS = 50
IMG_SIZE=64

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
        results = YOLO(file)
        object = results.crop(save=False)
        imgs = list()
        for obj in object:
            img = obj['im']
            removeexif = sp.removeexif(img)
            threshold = sp.threshold(removeexif)
            removebg = sp.removebg(threshold)
            mask = sp.mask(removebg)
            imgs.append((removeexif, sp.maskslic(img,mask,N_SEGMENTS)))
        
        return imgs

    
    def preprocess_segment(self,image,props,idx,size,filename):
        bbox = props.bbox
        props_image = Image.fromarray(image)
        cropped_img = props_image.crop(bbox)
        
        # resize image
        cropped_img = np.array(cropped_img)
        cropped_img = Image.fromarray(cropped_img)
        resized_img = cropped_img.resize((IMG_SIZE,IMG_SIZE))
        resized_img = np.array(resized_img)
        # resized_img = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2HSV)
        
        # save segments to check
        path = os.path.join(cfg['base']['path'],
                            cfg['image']['path'],
                            cfg['image']['inference']['path'],
                            cfg['image']['inference']['dir']['seg'])
        segment_img_file = os.path.join(path,'{:s}{:06d}.jpg'.format(filename,idx))
        cv2.imwrite(segment_img_file,resized_img)
        
        resized_img = np.expand_dims(resized_img, axis=0).astype(np.float32) / 255.0
        
        return resized_img
        

    def inference(self,image_name,image,slic):
        img=np.array(image)
        class_name = ['contaminated','uncontaminated']
        regions = regionprops(slic,intensity_image=img)
        contaminated=[]
        uncontaminated=[]
        size=len(slic)//math.sqrt(N_SEGMENTS)
        filename = image_name.split('/')[-1].split('.')[0]
        
        for idx,props in enumerate(regions):
            input_img = self.preprocess_segment(img,props,idx,size,filename)
            prediction = TENSORFLOW.predict(input_img)
            
            if class_name[np.argmax(prediction)]=='contaminated':
                contaminated.append(idx)
            else:
                uncontaminated.append(idx)
        
        print(filename)
        if len(contaminated) > len(uncontaminated):
            print('result: contaminated')
        else:
            print('result: uncontaminated')
        print("contaminated:",contaminated)
        print("uncontaminated:",uncontaminated)
        print()
    
    
    def process(self):
        glob_path = os.path.join(cfg['base']['path'],
                                 cfg['image']['path'],
                                 cfg['image']['inference']['path'],
                                 cfg['image']['inference']['dir']['raw'],'*')
        image_datas = glob(glob_path)

        for image_name in image_datas:
            imgs = self.preprocess_image(image_name)
            for img,slic in imgs:
                self.inference(image_name,img,slic)
    
    
    def __init__(self):
        self.use_gpu()
        self.process()
        

if __name__ == '__main__':
    Inference()
