import tensorflow as tf
import os
from config import cfg
import torch
import utility
import superpixel


YOLO = torch.hub.load('ultralytics/yolov5','custom', path=os.path.join(cfg['base']['path'],'last.pt'), force_reload=True)
N_SEGMENTS = 50


class Segment():

    def use_gpu(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in tf.config.experimental.list_physical_devices("GPU"):
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print("GPU used!")
            except RuntimeError as e:
                print(e)
                
                
    def process(self):
        path = os.path.join(cfg['base']['path'],
                            cfg['image']['path'],
                            cfg['image']['train']['path'],
                            cfg['image']['train']['dir']['raw'])
        
        files = utility.get_files_fullpath(path)
        
        for idx, file in enumerate(files):
            filename = os.path.basename(file)
            sp = superpixel.Superpixel()
            results = YOLO(file)
            # results.show()
            object = results.crop(save=False)
            imgs = list()
            
            for obj in object:
                img = obj['im']
                shape = img.shape
                if shape[0] <= 150 or shape[1] <=150:
                    continue
                removeexif = sp.removeexif(img)
                threshold = sp.threshold(removeexif)
                removebg = sp.removebg(threshold)
                mask = sp.mask(removebg)
                imgs.append((removebg, sp.maskslic(img,mask,N_SEGMENTS)))
            
            for img,slic in imgs:
                sp.saveslic(img,slic,filename)
            
            print('{} / {} {} Done.'.format(idx + 1, len(files), filename))
            # exit()

    def __init__(self):
        self.use_gpu()
        self.process()
    

if __name__ == '__main__':
    Segment()
