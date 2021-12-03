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


IMG_SIZE=64


class Superpixel:
    
    def removeexif(self, input_image):
        image = Image.fromarray(input_image)
        data = list(image.getdata())
        image_without_exif = Image.new(image.mode, image.size)
        image_without_exif.putdata(data)

        return image_without_exif


    def threshold(self, input_image):
        image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        thr_s = cv2.threshold(s, 92, 255, cv2.THRESH_BINARY)[1]
        thr_v = cv2.threshold(v, 128, 255, cv2.THRESH_BINARY)[1]
        thr_v = 255 - thr_v
        
        mask = cv2.add(thr_s, thr_v)
        image[mask==0] = (255, 255, 255)
        encoded_bytes = cv2.imencode('.jpeg', image)[1]
        
        return encoded_bytes
    
    
    def removebg(self, input_image):
        result = remove(input_image)
        image = Image.open(BytesIO(result)).convert("RGB")
        
        return image


    def mask(self, input_image):
        image = np.array(input_image)
        lum = color.rgb2gray(image)
        mask = morphology.remove_small_holes(morphology.remove_small_objects(lum > 0.05, 500), 500)
        mask = morphology.area_closing(mask)
        chull = morphology.convex_hull_image(mask)

        return chull


    def maskslic(self, input_image, mask, n_segments):
        image = np.array(input_image)
        m_slic = segmentation.slic(image, compactness=100, n_segments=n_segments, mask=mask, sigma=5, start_label=1)
        mask = color.gray2rgb(mask)
        masked_image = image*mask
        io.imsave('/home/ubuntu/dev/cordyceps/test/slic.jpg',segmentation.mark_boundaries(cv2.cvtColor(masked_image,cv2.COLOR_BGR2RGB),m_slic))
        
        return m_slic


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
            # cropped_img = np.array(cropped_img)
            cropped_img = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2HSV)
            
            path = os.path.join(cfg['base']['path'],
                                cfg['image']['path'],
                                cfg['image']['train']['path'],
                                cfg['image']['train']['dir']['seg'])
            segment_img_file = os.path.join(path,'{:03d}{:03d}.jpg'.format(index,idx))
            cv2.imwrite(segment_img_file,cropped_img)