from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import os
from rembg.bg import remove
from io import BytesIO
from skimage import color, morphology, segmentation, filters
from skimage.measure import regionprops
from config import cfg


class Superpixel:
    
    
    def removebg(self, input_image):
        result = remove(input_image)
        image = Image.open(BytesIO(result)).convert("RGB")
        
        return image


    def makeslic(self, input_image, n_segments):
        image = np.array(input_image)
        slic = segmentation.slic(image, n_segments=n_segments, sigma=5, start_label=1)
        
        return slic


    def saveslic(self, res, segments, index):
        image=np.array(res)
        regions = regionprops(segments,intensity_image=image)
        size=cfg['seg']['crop']

        for idx,props in enumerate(regions):
            cy, cx = props.centroid
            left = cx - int(size / 2)
            right = left + size
            top = cy - int(size / 2)
            bottom = top + size
            props_image = Image.fromarray(image)
            cropped_img = props_image.crop((left, top, right, bottom))
            
            cropped_img = np.array(cropped_img)
            cropped_img = Image.fromarray(cropped_img)
            cropped_img = cropped_img.resize((227,227))
            cropped_img = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2HSV)
            
            segment_img_file = os.path.join(cfg['seg']['path'],'{:03d}{:03d}.jpg'.format(index,idx))
            cv2.imwrite(segment_img_file,cropped_img)