import paho.mqtt.client as mqtt
from influxdb import InfluxDBClient
import datetime
import time
from absl import app
import json


# save data
def save_data(msg, receive_time, message):
    dbclient = InfluxDBClient('localhost', 8086, database='sensordata')
    json_data = json.loads(message)
    sensor_list=['co2','temperature','humidity','lux']
    message_list = list(json_data['value'].split(','))
    for i in range(len(message_list)):
        measurement = sensor_list[i]
        try:
            json_dic = [
                {
                    "measurement": measurement,
                    "time": receive_time,
                    "fields": {
                        "id" : str(json_data['id']),
                        "value" : float(message_list[i])
                    }
                }
            ]
            print(json_dic)
            dbclient.write_points(json_dic)
        except:
            pass


# connect
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("connection success!")
        client.subscribe('sensor/#')
    else:
        print("bad connection returned code = ", rc)


# disconnect
def on_disconnect(client, userdata, flags, rc=0):
    print("disconnection result code = ", str(rc))


# subscribe topic
def on_subscribe(client, userdata, mid, granted_qos):
    print("subscribed:", str(mid), str(granted_qos))


# receive message 
# save data
def on_message(client, userdata, msg):
    receive_time=datetime.datetime.utcnow()
    message=msg.payload.decode("utf-8")

    print("receive time: %s topic: %s message: %s" %(str(receive_time), msg.topic, str(message)))
   
    save_data(msg, receive_time, message)


def main(_argv):
    # create mqtt client
    client = mqtt.Client()

    # callback method
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_subscribe = on_subscribe
    client.on_disconnect = on_disconnect

    # address : gogogo.kr
    # port: 1883
    check_connection = False

    while(check_connection == False):
        try:
            client.connect('gogogo.kr', 1883, 60)
            check_connection = True
        except:
            check_connection = False
        time.sleep(5)

    client.loop_forever()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass