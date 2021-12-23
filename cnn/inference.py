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
from skimage import io
import json
import requests


YOLO = torch.hub.load('ultralytics/yolov5','custom', path=os.path.join(cfg['base']['path'],'last.pt'), force_reload=True)
TENSORFLOW = tf.keras.models.load_model(os.path.join(cfg['base']['path'],cfg['model']['dir'],cfg['model']['version']))
N_SEGMENTS = 100
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

    
    def preprocess_segment(self,image,cropped_img,idx,size,filename):
        # bbox = props.bbox
        # props_image = Image.fromarray(image)
        # cropped_img = props_image.crop(bbox)
        
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
        contaminated = 0
        size=len(slic)//math.sqrt(N_SEGMENTS)
        filename = image_name.split('/')[-1]
        origin_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        true_or_false = 'f'
        
        for idx,props in enumerate(regions):
            bbox = props.bbox
            props_image = Image.fromarray(img)
            cropped_img = props_image.crop(bbox)
            shape = np.array(cropped_img).shape
            if shape[0]/shape[1] > 1.5 or shape[1]/shape[0] > 1.5:
                continue
            
            input_img = self.preprocess_segment(img,cropped_img,idx,size,filename)
            img_tensor = input_img
            
            data = json.dumps({"signature_name": "serving_default", "instances": img_tensor.tolist()})
        
            # request prediction
            headers = {"content-type": "application/json"}
            json_response = requests.post('http://cordyceps@gogogo.kr:8501/v1/models/cordyceps_pretrain:predict', data=data, headers=headers)
            
            predictions = json.loads(json_response.text)['predictions']
            result = class_name[np.argmax(predictions[0])]
            confidence = int(np.max(predictions[0])*100)
            if result == 'contaminated' and confidence > 65:
                contaminated+=1
                minr, minc, maxr, maxc = props.bbox
                x,y,w,h = minr,minc,maxr-minr,maxc-minc
                origin_image = cv2.rectangle(origin_image,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(origin_image, str(confidence)+'%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, lineType=cv2.LINE_AA)
        
        io.imsave('/home/ubuntu/dev/cordyceps/process/result.jpg', origin_image)
        
        if contaminated/len(regions)>0.3:
            true_or_false = 't'
        print(true_or_false)
    
    
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
