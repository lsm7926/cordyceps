import tensorflow as tf
import numpy as np
import preprocess
from skimage.measure import regionprops
from PIL import Image
import os
from glob import glob
import cv2
from config import cfg

from logging.handlers import RotatingFileHandler
import logging
logger = logging.getLogger()


class Main():
    def job(self):
        logger.info('job')
    
    
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
        removeexif = preprocess.removeexif(file)
        
        # upscale image
        # image = np.array(removeexif)
        # sr = cv2.dnn_superres.DnnSuperResImpl_create()
        # path = os.path.join(cfg['model']['dir'],'EDSR_x4.pb')
        # sr.readModel(path)
        # sr.setModel("edsr", 10)
        # removeexif = Image.fromarray(sr.upsample(image))
        # print('upscale finish!')
        
        threshold = preprocess.threshold(removeexif)
        removebg = preprocess.removebg(threshold)
        mask = preprocess.mask(removebg)
        centroid = preprocess.center(mask)
        cropped_removeexif, cropped_mask = preprocess.crop(removeexif,mask,400,centroid)
        segments, masked_image = preprocess.maskslic(cropped_removeexif,cropped_mask,50)
        
        return segments, masked_image

    
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
        

    def inference(self,image_name,masked_image,segments):
        image=np.array(masked_image)
        class_name = ['contaminated','uncontaminated']
        regions = regionprops(segments,intensity_image=image)
        contaminated=[]
        uncontaminated=[]
        size=64
        
        # inference
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
            segments, masked_image = self.preprocess_image(image_name)
            self.inference(image_name,masked_image,segments)
    
    
    def __init__(self):
        logger.info('init')
        self.use_gpu()
        self.process()
        
        '''
        schedule.every(1).minutes.do(self.job)
        while True:
            schedule.run_pending()
        '''

if __name__ == '__main__':
    formatter = logging.Formatter('%(asctime)s %(message)s', "%Y-%m-%d %H:%M:%S")
    
    handler = RotatingFileHandler(os.path.join(cfg['base']['path'],'logger.log'), mode='a', maxBytes=3*1024*1024, backupCount=2)
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info('main')
    Main()
    logger.info('quit')
