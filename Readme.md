# AWS Serverless Object Detection and Computational Offloading

An end-to-end **cloud-based object detection service** demonstrating computational offloading from a local environment to AWS. The project implements both **local and remote object detection** using YOLOv3-tiny and compares their performance in terms of inference time, data-transfer overhead, scalability, and accuracy.

The cloud-based solution uses **Amazon S3, AWS Lambda, and Amazon DynamoDB** to create a serverless object detection pipeline.

---

## Overview

The project explores **computational offloading**, where computationally intensive tasks are transferred from a local machine to remote cloud infrastructure.

Two execution approaches are implemented:

1. **Local execution** using Flask and YOLOv3-tiny.
2. **Remote execution** using AWS S3, Lambda, and DynamoDB.

The two approaches are evaluated using the same object detection model and a dataset of 100 images.

---

## Architecture

### Local Execution

```text
Client
   │
   │ HTTP POST
   ▼
Flask Server
   │
   ▼
YOLOv3-tiny
   │
   ▼
Object Detection
   │
   ▼
JSON Response
```

### Cloud Execution

```text
Client
   │
   │ Upload Image
   ▼
Amazon S3
   │
   │ S3 Event
   ▼
AWS Lambda
   │
   ▼
YOLOv3-tiny
   │
   ▼
Object Detection
   │
   ▼
Amazon DynamoDB
   │
   ▼
Detection Results
```

---

## Key Features

* Local object detection using Flask
* Cloud-based object detection using AWS Lambda
* YOLOv3-tiny object detection model
* REST API for local image processing
* Base64 image encoding and decoding
* Amazon S3 image storage
* S3-triggered AWS Lambda execution
* Amazon DynamoDB result storage
* Performance benchmarking
* Local vs. cloud execution comparison

---

## Technologies

| Technology          | Purpose                             |
| ------------------- | ----------------------------------- |
| **Python**          | Application development             |
| **Flask**           | Local REST API                      |
| **OpenCV**          | Image processing and YOLO inference |
| **YOLOv3-tiny**     | Object detection                    |
| **AWS S3**          | Image storage                       |
| **AWS Lambda**      | Serverless object detection         |
| **Amazon DynamoDB** | Detection-result storage            |
| **boto3**           | AWS integration                     |
| **NumPy**           | Image data processing               |

---

## Local Execution

The local implementation uses a Flask-based REST service to receive images and perform object detection.

### Workflow

1. The client reads an image from the local dataset.
2. The image is encoded as a Base64 string.
3. The client sends the image to the Flask API using an HTTP POST request.
4. The Flask server decodes the image.
5. OpenCV processes the image using YOLOv3-tiny.
6. Detected objects and confidence scores are returned as a JSON response.

### API Endpoint

```text
POST /api/object_detection
```

The request contains:

* Base64-encoded image
* Unique image identifier

The response contains the detected objects and their confidence scores.

---

## Cloud Execution

The remote implementation offloads object detection to AWS.

### Workflow

1. The client uploads an image to an Amazon S3 bucket.
2. The S3 upload event triggers an AWS Lambda function.
3. Lambda retrieves the image from S3.
4. YOLOv3-tiny performs object detection.
5. Detected objects and confidence scores are generated.
6. The results are stored in Amazon DynamoDB.
7. The stored results can be retrieved for later use.

This architecture removes the need for the client machine to perform the computationally intensive inference locally.

---

## AWS Components

### Amazon S3

S3 is used to store uploaded images and acts as the entry point for the cloud processing pipeline.

### AWS Lambda

Lambda executes the object detection function when a new image is uploaded to S3.

### Amazon DynamoDB

DynamoDB stores:

* Image identifier
* Detected objects
* Confidence scores
* Processed image information
* Detection results

---

## Performance Evaluation

The system was evaluated by processing **100 images** in both local and remote environments.

### Experimental Setup

#### Local Environment

* Intel Core i7 CPU
* 16 GB RAM
* Flask
* OpenCV
* YOLOv3-tiny
* Python

#### Cloud Environment

* Amazon S3
* AWS Lambda
* Amazon DynamoDB
* YOLOv3-tiny

---

## Results

| Metric                   |  Local Execution |  Cloud Execution |
| ------------------------ | ---------------: | ---------------: |
| Average Inference Time   | **0.38 s/image** | **0.45 s/image** |
| Image Transfer Time      |                — | **0.10 s/image** |
| Average Confidence Score |         **0.81** |         **0.81** |

### Interpretation

The local implementation achieved a slightly lower inference time than the cloud implementation.

The additional latency in the cloud approach is primarily associated with:

* Image upload
* Network communication
* S3 processing
* Lambda invocation
* Cloud execution overhead

However, both approaches achieved a similar average confidence score of **0.81**, indicating that moving the computation to the cloud did not significantly affect detection accuracy.

---

## Local vs. Cloud

| Aspect                    | Local Execution           | Cloud Execution               |
| ------------------------- | ------------------------- | ----------------------------- |
| Compute Location          | Local machine             | AWS Lambda                    |
| Storage                   | Local                     | Amazon S3                     |
| Result Storage            | Local response            | DynamoDB                      |
| Inference                 | YOLOv3-tiny               | YOLOv3-tiny                   |
| Average Inference         | 0.38 s                    | 0.45 s                        |
| Scalability               | Limited by local hardware | Higher scalability            |
| Network Overhead          | Minimal                   | Present                       |
| Infrastructure Management | Required                  | Reduced                       |
| Best For                  | Small workloads           | Variable / scalable workloads |

---

## Advantages of Computational Offloading

The cloud-based approach provides several benefits:

### Scalability

AWS Lambda can support concurrent processing without requiring the client to provision additional local hardware.

### Resource Efficiency

Computationally intensive object detection is moved away from resource-constrained client machines.

### Flexible Infrastructure

The serverless architecture reduces the need to continuously manage dedicated servers.

### Persistent Results

Detection results can be stored in DynamoDB and retrieved independently of the original processing request.

---

## Limitations

The cloud implementation also introduces additional overhead:

* Network latency
* Image upload time
* Lambda invocation overhead
* AWS configuration requirements
* Dependency and model packaging constraints
* Potential cloud usage costs

For small workloads, local execution can therefore provide lower latency.

---

## Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Local Execution

Start the Flask server:

```bash
python app.py
```

Then use the client script to send images to the object detection endpoint:

```bash
python app_flask_client.py
```

### AWS Execution

The cloud workflow is:

```text
Image
  ↓
S3 Upload
  ↓
S3 Event Notification
  ↓
Lambda
  ↓
YOLOv3-tiny
  ↓
DynamoDB
```

---
