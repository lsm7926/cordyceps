import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from absl import app
from config import cfg


print(tf.__version__)
print(keras.__version__)
IMG_SIZE=227
BATCH_SIZE=16
SHUFFLE_BUFFER_SIZE = 1000
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


class Model:
    def __init__(self):
        self.dataset_path = os.path.join(cfg['base']['path'],
                                         cfg['data']['dir'],
                                         cfg['data']['dataset'])
        with open(self.dataset_path,"rb") as file:
            self.dataset = pickle.load(file)
        self.save_model_path = os.path.join(cfg['base']['path'],
                                            cfg['model']['dir'])
        self.train_images = self.dataset[0]  
        self.train_labels = self.dataset[1]
        self.test_images = self.dataset[2]
        self.test_labels = self.dataset[3]
        self.class_name = self.dataset[4]
        self.valid_images = None
        self.valid_labels = None
        self.model = None


    def use_gpu(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in tf.config.experimental.list_physical_devices("GPU"):
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print("GPU used!")
            except RuntimeError as e:
                print(e) 


    def make_directory(self, target_path):
        if not os.path.exists(target_path):
            os.mkdir(target_path)
            print('Directory {} is created'.format(target_path))
        else:
            print('Directory {} is already exists'.format(target_path))


    def preprocess_image(self):
        self.train_images = self.train_images.astype(np.float32) / 255.0
        self.test_images = self.test_images.astype(np.float32) / 255.0
        self.train_images, self.valid_images, self.train_labels, self.valid_labels = train_test_split(
            self.train_images, self.train_labels, stratify=self.train_labels, test_size=0.2, random_state=44
            )


    def setup_model(self):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights='imagenet')
        base_model.trainable = False
        base_model.summary()
        
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(len(self.class_name),activation='softmax')

        self.model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            prediction_layer
            ])
        
        # metrics >> accuracy, sparse_categorical_accuracy
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        # base_learning_rate = 0.0001
        # model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
        #             loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        #             metrics=['accuracy'])

        epochs = 20
        batch_size = 32
        is_callback = True
        checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(cfg['base']['path'],cfg['model']['callback']))
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

        self.model.fit(
            self.train_images,
            self.train_labels,
            epochs = epochs,
            validation_data=(self.valid_images, self.valid_labels),
            batch_size = batch_size,
            steps_per_epoch=self.train_images.shape[0]//batch_size,
            callbacks=[checkpoint_cb, early_stopping_cb] if is_callback else None
        )


    def save_model(self):
        export_path = os.path.join(self.save_model_path, cfg['model']['version'])

        tf.keras.models.save_model(
            self.model,
            export_path,
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None
        )

        print('Model saved to : {}'.format(export_path))


def main(_argv):
    model = Model()
    model.preprocess_image()
    model.setup_model()
    model.save_model()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass