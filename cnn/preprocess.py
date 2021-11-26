from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
from rembg.bg import remove
from io import BytesIO
from skimage import color, morphology, segmentation, filters
from skimage.measure import regionprops


def removeexif(input_image):
    image = Image.open(input_image)
    data = list(image.getdata())
    image_without_exif = Image.new(image.mode, image.size)
    image_without_exif.putdata(data)
    
    return image_without_exif


def threshold(input_image):
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


def removebg(input_image):
    result = remove(input_image)
    image = Image.open(BytesIO(result)).convert("RGB")
    
    return image


def mask(input_image):
    image = np.array(input_image)
    lum = color.rgb2gray(image)
    mask = morphology.remove_small_holes(morphology.remove_small_objects(lum > 0.05, 500), 500)
    mask = morphology.area_closing(mask)
    chull = morphology.convex_hull_image(mask)

    return chull


def maskslic(input_image, mask, n_segments):
    image = np.array(input_image)
    m_slic = segmentation.slic(image, n_segments=n_segments, mask=mask, sigma=5, start_label=1)
    mask = color.gray2rgb(mask)
    masked_image = image*mask

    return m_slic, masked_image


def center(input_image):
    image = color.rgb2gray(input_image)
    threshold_value = filters.threshold_otsu(image)
    labeled_foreground = (image > threshold_value).astype(int)
    properties = regionprops(labeled_foreground, image)
    center_of_mass = properties[0].centroid

    return center_of_mass


def crop(input_image_removeexif, input_image_mask, size, center):
    left = center[1] - int(size / 2)
    right = left + size
    top = center[0] - int(size / 2)
    bottom = top + size
    
    cropped_image_removeexif = input_image_removeexif.crop((left, top, right, bottom))
    input_image_mask = Image.fromarray(input_image_mask)
    cropped_image_mask = morphology.convex_hull_image((np.asarray(input_image_mask.crop((left, top, right, bottom)))))
    
    return cropped_image_removeexif, cropped_image_mask