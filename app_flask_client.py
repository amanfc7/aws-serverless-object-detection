import sys
import os
import uuid
import time
import json
import requests
import base64
import argparse

# function for encoding an image file to base64 format:

def img_encoding(path_img):
    with open(path_img, "rb") as img_file:
        encoding = base64.b64encode(img_file.read()).decode('utf-8')
    return encoding

# main function to process images from a folder and send them to an endpoint:

def main(input_folder, endpoint):
    stime = time.time()
    responses = []

    # to I=iterate over files in the input folder:

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path_img = os.path.join(input_folder, filename)
            image_id = str(uuid.uuid4())
            encode_img = img_encoding(path_img)

            # to prepare payload with image metadata and encoded image data:

            payload = {
                "image_name": filename,
                "id": image_id,
                "image_data": encode_img
            }

            # for sending POST request to the endpoint with JSON payload:

            response = requests.post(endpoint, json=payload)

            # Processing response
            if response.status_code == 200:
                response_img = response.json()
                responses.append(response_img)
                print(response_img)
            else:
                print(f"Cannot process the file: {filename}, status code: {response.status_code}")

    ending_time = time.time()

    # calculation of execution time:

    execution_time = ending_time - stime
    print(f"\nTotal time: {execution_time} seconds")

    # for saving all responses to a JSON file:

    with open("Total_output_and_results.json", "w") as outputfile:
        json.dump(responses, outputfile, indent=4)

if __name__ == "__main__":

    # Now,using argparse to handle command-line arguments:

    parser = argparse.ArgumentParser(description="Process images from a folder and send them to an endpoint")
    parser.add_argument('input_folder', type=str, help='Path to the folder containing images')
    parser.add_argument('endpoint', type=str, help='Endpoint URL to send images for processing')

    args = parser.parse_args()

    # for input folder and endpoint from command-line arguments: 

    input_folder = args.input_folder
    endpoint = args.endpoint
    
    # Calling the main function with provided arguments:
    
    main(input_folder, endpoint)
