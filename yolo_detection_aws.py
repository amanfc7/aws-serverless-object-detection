import base64
import os
import numpy as np
import json
import cv2
import uuid
import boto3

# to initialize AWS services:

s3 = boto3.client('s3')
dynamoDB = boto3.resource('dynamodb')
table = dynamoDB.Table('YOLOresult')


# loading YOLO configuration and weights:

yolo_net = cv2.dnn.readNet("/yolov3-tiny.weights", "/yolov3-tiny.cfg")
yolo_layer_names = yolo_net.getLayerNames()
output_layer_indices = yolo_net.getUnconnectedOutLayers()

# Determine YOLO output layers:

if isinstance(output_layer_indices[0], list):
    layer_out = [yolo_layer_names[i[0] - 1] for i in output_layer_indices]
else:
    layer_out = [yolo_layer_names[i - 1] for i in output_layer_indices]

# to load class labels for YOLO:

with open("/coco.names", "r") as f:
    label_class = [line.strip() for line in f.readlines()]

# Lambda handler function to process images from S3 and store results in DynamoDB:

def process_image_handler(event, context):
    
    # Extracting bucket name and object key from the S3 event:

    bucket_name = 'image-detector-dic-2024'
    object_key = event['Records'][0]['S3']['object']['key']

    # Retrieving the image data from S3:

    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    img_data = response['Body'].read()

    # Decoding image data using OpenCV:

    np_image_array = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_image_array, cv2.IMREAD_COLOR)

    image_height, image_width, image_channels = image.shape
    image_blob = cv2.dnn.blobFromImage(image, 0.00458, (500, 500), (0, 0, 0), True, crop=False)
    yolo_net.setInput(image_blob)
    yolo_output = yolo_net.forward(layer_out)

    # Parse detections and filter by confidence threshold:

    detected_class_id = []
    detection_confidences = []
    detection_boxes = []

    for output in yolo_output:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * image_width)
                center_y = int(detection[1] * image_height)
                width = int(detection[2] * image_width)
                height = int(detection[3] * image_height)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                detection_boxes.append([x, y, width, height])
                detection_confidences.append(float(confidence))
                detected_class_id.append(class_id)

    detected_objects_list = []
    indices = cv2.dnn.NMSBoxes(detection_boxes, detection_confidences, 0.5, 0.4)
    for i in range(len(detection_boxes)):
        if i in indices:
            box = detection_boxes[i]
            label = str(label_class[detected_class_id[i]])
            accuracy = detection_confidences[i]
            detected_objects_list.append({"label": label, "accuracy": accuracy})

    # the result for DynamoDB and to store it:

    result_objects = {
        "image_name": object_key,
        "id": str(uuid.uuid4()),
        "objects": detected_objects_list
    }

    dynamoDB.put_item(Item=result_objects)

    return {
        'statusCode': 200,
        'body': json.dumps(result_objects)
    }
