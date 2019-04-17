#############################################################
#															#
# Input   : (Currently) Reads image path from command line	#
#			(Ideally:)										#
#			Image of shape (batch_size) 3 * 416 * 416		#
#															#
# Returns : Output of 58th convolutional layer of yolov3.	#
#			i.e. layer 'conv_80'							#
#			(Currently) shape- 1 * 13 * 13 * 1024			#
#			(Ideally) shape- (batch_size) 13 * 13 * 1024	#
#															#
#############################################################

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

Width  = 416       #Width of network's input image
Height = 416      #Height of network's input image

parser = argparse.ArgumentParser()
parser.add_argument('--image')
args = parser.parse_args()
        
# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

if not os.path.isfile(args.image):
    print("Input image file ", args.image, " doesn't exist")
    sys.exit(1)
cap = cv.VideoCapture(args.image)

hasFrame, frame = cap.read()

# If: Input shape == 3*416*416: No need to create blob
# Else: Create a 4D blob from a frame.
blob = cv.dnn.blobFromImage(frame, 1/255, (Width, Height), [0,0,0], 1, crop=False)

# Sets the input to the network
net.setInput(blob)

# Runs the forward pass to get output of the output layers
layersNames = net.getLayerNames()
out = net.forward('conv_80')