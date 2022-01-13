import paho.mqtt.client as mqtt
import psycopg2
import datetime
import time
import json
import os
import torch
import tensorflow as tf
from config import cfg
from superpixel import Superpixel


YOLO = torch.hub.load('ultralytics/yolov5','custom', path=os.path.join(cfg['base']['path'],'last.pt'), force_reload=True)
TENSORFLOW = tf.keras.models.load_model(os.path.join(cfg['base']['path'],cfg['model']['dir'],cfg['model']['version']))
SERVER_PATH = '/media/image'

class Main():
    def __init__(self):
        self.superpixel = Superpixel()

        # create mqtt client
        self.mqtt_client = mqtt.Client()

        # callback method
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_subscribe = self.on_subscribe
        self.mqtt_client.on_disconnect = self.on_disconnect

        # loop start
        self.start_loop()


    def start_loop(self):
        # address : gogogo.kr
        # port: 1883
        check_connection = False

        while(check_connection == False):
            try:
                self.mqtt_client.connect('gogogo.kr', 1883, 60)
                check_connection = True
            except:
                check_connection = False
            time.sleep(5)

        self.mqtt_client.loop_forever()


    # connect
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("connection success!")
            self.mqtt_client.subscribe('image')
        else:
            print("bad connection returned code = ", rc)


    # disconnect
    def on_disconnect(self, client, userdata, flags, rc=0):
        print("disconnection result code = ", str(rc))


    # subscribe topic
    def on_subscribe(self, client, userdata, mid, granted_qos):
        print("subscribed:", str(mid), str(granted_qos))


    # receive message 
    # preprocess and inference image
    def on_message(self, client, userdata, msg):
        receive_time=datetime.datetime.now()
        message=msg.payload.decode("utf-8")
        json_data = json.loads(message)
        device_id = json_data['id']
        file_path = json_data['value']

        folder_name = json_data['value'].split('/')[-2]
        file_name = json_data['value'].split('/')[-1]

        print("receive time: %s topic: %s message: %s" %(str(receive_time), msg.topic, str(message)))
        
        result_list = self.superpixel.process(file_path,YOLO)
        for result in result_list:
            file_name, true_or_false, score = result
            self.save_data(file_name, receive_time, device_id, os.path.join(SERVER_PATH, folder_name, file_name), true_or_false, score)


    # save result of inference
    def save_data(self,file_name, receive_time, device_id, file_path, is_contaminated, score):
        db_cfg = cfg['database']
        con = psycopg2.connect(host=db_cfg['host'], dbname=db_cfg['dbname'], user=db_cfg['user'], password=db_cfg['password'], port=db_cfg['port'])
        cur = con.cursor()
        cur.execute(
            "INSERT INTO engine_image (file_name, time_stamp, device_id, file_path, is_contaminated, score) VALUES (%s, %s, %s, %s, %s, %s)",
            (file_name, str(receive_time), device_id, file_path, is_contaminated, score)
        )
        
        con.commit()
        cur.close()
        con.close()


if __name__ == '__main__':
    Main()