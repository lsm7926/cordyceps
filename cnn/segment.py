import tensorflow as tf
import os
from config import cfg
import torch
import utility
import superpixel

model = torch.hub.load('ultralytics/yolov5','custom', path='/home/ubuntu/dev/cordyceps/cnn/last.pt', force_reload=True)
n_segments = 25


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
        files = utility.get_files_fullpath(cfg['raw']['path'])
        for idx, file in enumerate(files):
            filename = os.path.basename(file)
            sp = superpixel.Superpixel()
            results = model(file)
            object = results.crop(save=False)
            image = list()
            for obj in object:
                image.append((obj['im'], sp.makeslic(obj['im'],n_segments)))
            
            for img,slic in image:
                slic = sp.makeslic(img,n_segments)
                sp.saveslic(img,slic,idx)
            
            print('{} / {} {} Done.'.format(idx + 1, len(files), filename))


    def __init__(self):
        self.use_gpu()
        self.process()
    

if __name__ == '__main__':
    Segment()
