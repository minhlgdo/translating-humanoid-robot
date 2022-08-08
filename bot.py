#!/usr/bin/env python
#
# First Steps in Programming a Humanoid AI Robot
#
# Text detection and recognition example code
# Detects texts and decodes them
#
# Import required modules
from pickle import TRUE
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('..')
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

from lib.ros_environment import ROSEnvironment
from lib.camera_v2 import Camera
from imutils.object_detection import non_max_suppression
import argparse

from PIL import Image
import pyocr
import pyocr.builders
import numpy as np
import translate
import time

#Check everything is installed
#We check what is installed and the available language
avail_tools = pyocr.get_available_tools()
avail_tools = avail_tools[0]
print("Available Tools: ", avail_tools.get_name())
print("Available Language: ", avail_tools.get_available_languages())


def main():
    # We need to initalize ROS environment for Robot and camera to connect/communicate
    ROSEnvironment()

    #Initialize and start camera
    camera = Camera()
    camera.start()

    # Load the network with a pretrained text detector
    # You can read about it in https://github.com/argman/EAST
    net = cv2.dnn.readNet("./frozen_east_text_detection.pb")
    # translate = goslate.Goslate()

    counter = 1
    # next_word = True
    closeWindows = False
    limit = 4

    exit = False
    # Loop
    while True:

        # Get image from camera and making copies
        img = camera.getImage()
        img_copy = img.copy()
        img_output = img.copy()

        # Change the size of the image
        # Smaller images are faster, but might be less accurate
        newWidth = 320
        newHeight = 320
        originWidth = 640
        originHeight = 480
        ratioWidth = originWidth / float(newWidth)
        ratioHeight = originHeight / float(newHeight)
        img_copy = cv2.resize(img_copy, (newWidth, newHeight))

        # Select the layer that we will be using. Conv_7/Sigmoid is used to get the probability.
        # concat_3 is used to determine the bounding box,. 
        layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

        """
        # identify all output layers (depend on network: YOLOv3 has 3, YOLOv3-tiny has 2)
        layer_names = net.getLayerNames()
        output_layers = [ layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers() ]

        # print the names of all layers and, separately, all output layers
        print("Network information:\n"
              "  layer names:\n    {}"
              "  output layers:\n    {}"
              "\n"
              .format(layer_names, output_layers))
        """


        # Create blobs that will be forward passed in the network
        blob = cv2.dnn.blobFromImage(img_copy, 1.0, (newWidth,newHeight), (123.68, 116.78, 103.94), swapRB=True, crop=False)

        # Input the blobs and forward pass the blobs
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)

        # The postProcess method takes in scores, geometry, and minimum score. If the score is less than the minimum score, it is removed.
        # The postProcess method outputs the bounding boxes (x, y, width, height) and the confidence scores
        (bboxes, confidences) = postProcess(scores, geometry, 0.9)

        # Removes overlapping boxes
        bboxes = non_max_suppression(np.array(bboxes), probs=confidences)

        # Draw rectangles using the bounding boxes
        if cv2.waitKey(1) == ord('c'):
            for (startPtX, startPtY, endPtX, endPtY) in bboxes:
                # Need to consider the resizing and the cv2.rectangle takes in start points and end points. So, need to compute the start points and the end points
                startPtX = int(startPtX * ratioWidth)
                startPtY = int(startPtY * ratioHeight)
                endPtX = int(endPtX * ratioWidth)
                endPtY = int(endPtY * ratioHeight)

                # For the detected text and its bounding box, we want to recognize the text. We will create some buffer around the text and make sure it doesn't go over the image size.
                (startPtX, startPtY, endPtX, endPtY) = createBuffer(startPtX, startPtY, endPtX, endPtY)

                # We cut the region in the image where we want to recognize text
                roi = img[startPtY:endPtY, startPtX:endPtX, :] # ROI is short for region of interest

                # Convert the format of the image
                img_pil = Image.fromarray(roi)

                # Use pyocr to recognize the text in the roi image

                # print("123")
                text = avail_tools.image_to_string(img_pil, lang='eng', builder=pyocr.builders.TextBuilder())


                # Remove non-ascii characters with a space
                text = "".join([char if ord(char) < 128 else "" for char in text]).strip()

                # Print recognized text
                print("{}\n".format(text))
                translated = ""

                if text != "":
                    print("Count iteration: {}/{}".format(counter, limit))
                    print("Check: " + text)
                    translated = translate.translate(text)
                    # counter += 1

                    if counter == limit:
                        exit = True
                    
                    counter += 1
                        
                    # translated_word = translate.translate(text, 'de')
                    # print(transslated_word)
                # print(translate.translate(text, 'de'))

                # Print the text on the image
                cv2.putText(img_output, translated, (startPtX, startPtY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.rectangle(img_output, (startPtX, startPtY), (endPtX, endPtY), (0, 255, 255), 2)
                
            
        # Show image
        cv2.imshow("Frame2", img_output[...,::-1])
        # time.sleep(10)

        # Close if key a is pressed to break
        if cv2.waitKey(33) == ord('a') or exit:
            break


def postProcess(scores, geometry, minConfidence):
    # Go through the row and column to find the bounding box and confidence score
    (rows, cols) = scores.shape[2:4]
    bboxes = []
    confidence_scores = []

    for rowIndex in range(0, rows):
        score = scores[0, 0, rowIndex]
        # Need to use geometric data to determine the bounding box
        geoData0 = geometry[0, 0, rowIndex]
        geoData1 = geometry[0, 1, rowIndex]
        geoData2 = geometry[0, 2, rowIndex]
        geoData3 = geometry[0, 3, rowIndex]
        angleData = geometry[0, 4, rowIndex]

        for colIndex in range(0, cols):
            # If the confidence score is too low, we can skip
            if score[colIndex] < minConfidence:
                continue
            # Need to find the offset because the feature map is 4 times smaller than the original image size
            (xOffset, yOffset) = (colIndex * 4.0, rowIndex * 4.0)

            # Get the rotation angle
            angle = angleData[colIndex]
            # Compute cos and sin of the rotation angle
            cosAngle = np.cos(angle)
            sinAngle = np.sin(angle)

            # Compute the width and height
            width = geoData1[colIndex] + geoData3[colIndex]
            height = geoData0[colIndex] + geoData2[colIndex]

            endPtX = int(xOffset + (cosAngle * geoData1[colIndex]) + (sinAngle * geoData2[colIndex]))
            endPtY = int(yOffset - (sinAngle * geoData1[colIndex]) + (cosAngle * geoData2[colIndex]))
            startPtX = int(endPtX - width)
            startPtY = int(endPtY - height)
            bboxes.append((startPtX, startPtY, endPtX, endPtY))
            confidence_scores.append(score[colIndex])
    return bboxes, confidence_scores

# The bounding box from the text detector might be too small so that it cuts some of the text
# We want to create a bigger bounding box but without going over the border
def createBuffer(startX, startY, endX, endY):
    startX = startX -5
    endX = endX+5
    startY = startY -5
    endY = endY +5
    if(startX < 0):
        startX = 1
    if(startX > 640):
        startX = 640
    if(endX < 0):
        endX = 1
    if(endX > 640):
        endX = 640
    if(startY < 0):
        startY = 1
    if(startY > 480):
        startY = 480
    if(endY < 0):
        endY = 1
    if(endY > 480):
        endY = 480
    return startX, startY, endX, endY
if __name__ == '__main__':
    main()
