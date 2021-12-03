import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
from glob import glob
from PIL import Image, ImageFile
import os
from absl import app
import pickle
import matplotlib.pyplot as plt
import time
ImageFile.LOAD_TRUNCATED_IMAGES = True
from config import cfg

print(tf.__version__)
print(keras.__version__)


IMG_SIZE=64


class Dataset():
    def __init__(self):
        self.base_path = os.path.join(cfg['base']['path'],cfg['data']['dir'])
        self.image_data = list()
        self.label_data = list()
        self.class_name = None
        self.train_images = None
        self.test_images = None
        self.train_labels = None
        self.test_labels = None

    
    def get_class_name(self):
        label_path = os.path.join(self.base_path, cfg['data']['label'])
        with open(label_path,"r") as file:
            class_name = file.read().splitlines()
        return class_name
    

    def generate_dataset(self):
        self.class_name = self.get_class_name()
        
        for label in self.class_name:
            image_path = os.path.join(self.base_path,label)
            glob_path = os.path.join(image_path,'*')
            image_datas = glob(glob_path)
            
            for image_name in image_datas:
                image = Image.open(image_name)
                image = image.resize((IMG_SIZE,IMG_SIZE))
                image = np.array(image)
                self.image_data.append(image)
                self.label_data.append(self.class_name.index(label))
        
        self.image_data = np.array(self.image_data)
        self.label_data = np.array(self.label_data)
        self.train_images, self.test_images, self.train_labels, self.test_labels = train_test_split(
            self.image_data, self.label_data, test_size=0.2, shuffle=True, random_state=44
            )

        self.train_labels = self.train_labels[..., tf.newaxis]
        self.test_labels = self.test_labels[..., tf.newaxis]


    def save_dataset(self):
        data = [self.train_images, self.train_labels, self.test_images, self.test_labels, self.class_name]
        with open(os.path.join(self.base_path,cfg['data']['dataset']), 'wb') as file:
          pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
        print('Complete!')


    def show_img(self):
        plt.figure(figsize=(15,9))
        for i in range(15):
            plt.subplot(3,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.train_images[i])
            plt.xlabel(self.class_name[self.train_labels[i][0]])


def main(_argv):
    start_time = time.time()
    dataset = Dataset()
    dataset.generate_dataset()
    dataset.save_dataset()
    end_time = time.time()
    print(end_time-start_time)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass