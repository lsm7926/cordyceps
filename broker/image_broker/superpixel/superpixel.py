from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import os
from rembg.bg import remove
from io import BytesIO
from skimage import io, color, morphology, segmentation
from skimage.measure import regionprops
from config import cfg
import math
import json
import requests
# from yolov5 import detect

N_SEGMENTS = 25
IMG_SIZE=64


class Superpixel():
    
    # removing exif data
    def removeexif(self, input_image):
        image = Image.fromarray(input_image)
        data = list(image.getdata())
        image_without_exif = Image.new(image.mode, image.size)
        image_without_exif.putdata(data)

        return image_without_exif


    # image thresholding
    def threshold(self, input_image):
        image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        thr_s = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY)[1]
        thr_v = cv2.threshold(v, 128, 255, cv2.THRESH_BINARY)[1]
        thr_v = 255 - thr_v
        
        mask = cv2.add(thr_s, thr_v)
        image[mask==0] = (255, 255, 255)
        encoded_bytes = cv2.imencode('.jpg', image)[1]
        
        return encoded_bytes
    
    
    # removing background
    def removebg(self, input_image):
        result = remove(input_image)
        image = Image.open(BytesIO(result)).convert("RGB")
        
        return image


    # masking
    def mask(self, input_image):
        image = np.array(input_image)
        lum = color.rgb2gray(image)
        mask = morphology.remove_small_holes(morphology.remove_small_objects(lum > 0.05, 500), 500)
        mask = morphology.area_closing(mask)
        chull = morphology.convex_hull_image(mask)

        return chull


    # image segmentation
    def maskslic(self, input_image, mask, n_segments):
        image = np.array(input_image)
        m_slic = segmentation.slic(image, compactness=10, n_segments=n_segments, mask=mask, sigma=5, start_label=1)
        
        return m_slic


    # detect and preprocess object
    def preprocess_image(self,file,model):
        results = model(file)
        object = results.crop(save=False)
        
        imgs = list()
        for obj in object:
            img = obj['im']
            shape = img.shape
            if shape[0] <= 150 or shape[1] <=150:
                continue
            removeexif = self.removeexif(img)
            threshold = self.threshold(removeexif)
            removebg = self.removebg(threshold)
            mask = self.mask(removebg)
            imgs.append((removebg, self.maskslic(img,mask,N_SEGMENTS)))
        
        return imgs


    # preprocess segment
    def preprocess_segment(self,image,props,idx,size,filename):
        bbox = props.bbox
        props_image = Image.fromarray(image)
        cropped_img = props_image.crop(bbox)
        
        # resize image
        cropped_img = np.array(cropped_img)
        cropped_img = Image.fromarray(cropped_img)
        resized_img = cropped_img.resize((IMG_SIZE,IMG_SIZE))
        resized_img = np.array(resized_img)
        
        # saving segments to check
        path = os.path.join(cfg['base']['path'],
                            cfg['image']['path'],
                            cfg['image']['inference']['path'],
                            cfg['image']['inference']['dir']['seg'])
        segment_img_file = os.path.join(path,'{:s}{:06d}.jpg'.format(filename,idx))
        cv2.imwrite(segment_img_file,resized_img)
        
        resized_img = np.expand_dims(resized_img, axis=0).astype(np.float32) / 255.0
        
        return resized_img
        

    # inference segments
    def inference(self,image_name,image,slic):
        img = np.array(image)
        class_name = ['contaminated','uncontaminated']
        regions = regionprops(slic,intensity_image=img)
        contaminated = 0
        size=len(slic)//math.sqrt(N_SEGMENTS)
        filename = image_name.split('/')[-1]
        origin_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        true_or_false = 'f'
        score = 0
        
        for idx,props in enumerate(regions):
            input_img = self.preprocess_segment(img,props,idx,size,filename)
            img_tensor = input_img
            
            data = json.dumps({"signature_name": "serving_default", "instances": img_tensor.tolist()})
        
            # request prediction
            headers = {"content-type": "application/json"}
            json_response = requests.post(cfg['model']['url'], data=data, headers=headers)
            
            predictions = json.loads(json_response.text)['predictions']
            result = class_name[np.argmax(predictions[0])]
            confidence = int(np.max(predictions[0])*100)
            
            if result == 'contaminated':
                contaminated+=1
                score += confidence
                minr, minc, maxr, maxc = props.bbox
                x,y,w,h = minr,minc,maxr-minr,maxc-minc
                origin_image = cv2.rectangle(origin_image,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(origin_image, str(confidence)+'%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, lineType=cv2.LINE_AA)
        
        io.imsave(image_name, origin_image)
        
        if contaminated/len(regions)>0.3:
            true_or_false = 't'
            
        return(filename, true_or_false, score)


    # main process
    def process(self,image_name,model):
        result_list = list()
        imgs = self.preprocess_image(image_name,model)
        
        for img,slic in imgs:
            result = self.inference(image_name,img,slic)
            result_list.append(result)
            
        return result_list


    # saving segments
    def saveslic(self, res, segments, index):
        image=np.array(res)
        regions = regionprops(segments,intensity_image=image)

        for idx,props in enumerate(regions):
            bbox = props.bbox
            props_image = Image.fromarray(image)
            cropped_img = props_image.crop(bbox)
            
            cropped_img = np.array(cropped_img)
            cropped_img = Image.fromarray(cropped_img)
            cropped_img = cropped_img.resize((IMG_SIZE,IMG_SIZE))
            cropped_img = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2HSV)
            
            path = os.path.join(cfg['base']['path'],
                                cfg['image']['path'],
                                cfg['image']['train']['path'],
                                cfg['image']['train']['dir']['seg'])
            segment_img_file = os.path.join(path,'{:03d}{:03d}.jpg'.format(index,idx))
            cv2.imwrite(segment_img_file,cropped_img)