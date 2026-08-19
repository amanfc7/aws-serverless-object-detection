import cv2
import numpy as np
import uuid
import base64
import flask
from flask import Flask, request, jsonify
import psutil 
import time 


app = Flask(__name__)

# Path to YOLO files:
config_path = "/Group 21_DIC2024_Ex3/Object-Detection-YOLO/yolov3-tiny.cfg"
weights_path = "/Group 21_DIC2024_Ex3/Object-Detection-YOLO/yolov3-tiny.weights"
names_path = "/Group 21_DIC2024_Ex3/Object-Detection-YOLO/coco.names"

# loading of YOLO files:

net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
layers = net.getLayerNames()
output_index = net.getUnconnectedOutLayers()

# checking for indices compatibility:

if isinstance(output_index[0], list):
    layer_out = [layers[i[0] - 1] for i in output_index]
else:
    layer_out = [layers[i - 1] for i in output_index]

# for loading COCO labels:

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Lists to store times for performance measurement:

system_time = []
cpu_time = []
execution_time = []

@app.route('/api/object_detection', methods=['POST'])
def detect_objects():

    # starting time measurements:
    stime = time.time()
    stime_system = psutil.Process().cpu_times().system
    stime_cpu = time.process_time()

# Parsing of incoming data:

    data = request.json
    id = data['id']
    name = data['image_name']
    data = base64.b64decode(data['image_data'])
    npr = np.frombuffer(data, np.uint8)
    images = cv2.imdecode(npr, cv2.IMREAD_COLOR)


    width, height = images.shape   # for getting image dimensions:

   # Preparing the image for YOLO:

    dim = cv2.dnn.blobFromImage(images, 0.00543, (500, 500), (0, 0, 0), True, crop=False)
    res = net.forward(layer_out)
    net.setInput(dim)
    
# to initialize lists for detected object attributes:

    id_att = []
    surity = []
    tags = []

   # parsing the outputs:

    for rest in res:
        for detect in rest:
            scores = detect[5:]
            id_att = np.argmax(scores)
            surity = scores[id_att]
            if surity > 0.5:
                c_c = int(detect[0] * width)
                c_d = int(detect[1] * height)
                a = int(detect[2] * width)
                b = int(detect[3] * height)
                c = int(c_c - a / 2)
                d = int(c_d - b / 2)
                tags.append([c, d, a, b])
                surity.append(float(surity))
                id_att.append(id_att)
    
     # filtering overlapping boxes(tags) using non-max suppression:

    detection = []
    index = cv2.dnn.NMSBoxes(tags, surity, 0.5, 0.4)
    for i in range(len(tags)):
        if i in index:
            tag = tags[i]
            label = str(classes[id_att[i]])
            accuracy = surity[i]
            detection.append({"label": label, "accuracy": accuracy})

   # measurements for Ending times: 

    t_ending = time.time()
    tcpu_ending = time.process_time()
    tsys_ending = psutil.Process().cpu_times().system

    # Now, for calculating execution time:

    total_time = t_ending - stime
    execution_time.append(total_time)

    # calculation of user CPU time and system CPU time:

    cpu_time_ = tcpu_ending - stime_cpu
    cpu_time.append(cpu_time_)
    system_time_ = tsys_ending - stime_system
    system_time.append(system_time_)

# Preparation of the response:

    response = {"image_name": name, "id": data['id'], "objects": detection}
    return jsonify(response)

@app.after_request
def after_request(response):
    global execution_time, cpu_time, system_time

    # calculation of average times:

    
    if len(cpu_time) > 0:
        average_cpu_time = sum(cpu_time) / len(cpu_time)
    else:
        average_cpu_time = 0.0

    if len(execution_time) > 0:
        average_execution_time = sum(execution_time) / len(execution_time)
    else:
        average_execution_time = 0.0

    if len(system_time) > 0:
        average_system_time = sum(system_time) / len(system_time)
    else:
        average_system_time = 0.0    
    

   # print the output result for the average of time:

    print(f"Averagw Time: {average_execution_time} seconds")
    print(f"Avg User CPU Time: {average_cpu_time} seconds")
    print(f"Avg System CPU Time: {average_system_time} seconds")

    # to reset accumulated times for the next request:

    execution_time = []
    cpu_time = []
    system_time = []

    return response

if __name__ == '__main__':

    # running the Flask app
    app.run(port=5000, debug=True)
