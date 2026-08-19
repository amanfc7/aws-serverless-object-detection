import json
import base64
import uuid
import boto3
import cv2
import numpy as np
import os

# Initialize AWS services
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('YOLOResults')

# Load YOLO files (make sure these files are available in the Lambda deployment package)
net = cv2.dnn.readNet("/var/task/yolov3-tiny.weights", "/var/task/yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
if isinstance(output_layers_indices[0], list):
    output_layers = [layer_names[i[0] - 1] for i in output_layers_indices]
else:
    output_layers = [layer_names[i - 1] for i in output_layers_indices]

with open("/var/task/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


def lambda_handler(event, context):
    """
    Lambda function handler which is responsible for processing images from S3 using YOLO for object detection,
    and store results in DynamoDB.

    Arguments:
        event (dict): Event data passed to the Lambda function.
                      Expected structure: {'Records': [{'s3': {'object': {'key': 'object_key'}}}]}
        context (object): Lambda Context runtime methods and attributes.

    Returns:
        dict: A dictionary containing the HTTP status code and the result JSON string.
    """

    # Extract bucket name and object key from S3 event
    bucket_name = 'image-detector-dic-2024'
    key = event['Records'][0]['s3']['object']['key']

    # Retrieve image data from S3
    response = s3.get_object(Bucket=bucket_name, Key=key)
    image_data = response['Body'].read()

    # Decode image data and process with OpenCV
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Parse detections and filter by confidence threshold
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    detected_objects = []
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            box = boxes[i]
            label = str(classes[class_ids[i]])
            accuracy = confidences[i]
            detected_objects.append({"label": label, "accuracy": accuracy})

    
    # Prepare result JSON for DynamoDB and store the item
    result = {
        "image_name": key,
        "id": str(uuid.uuid4()),
        "objects": detected_objects
    }

    table.put_item(Item=result)

    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
