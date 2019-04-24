#################################################################
#                                                               #
# Input   : (Currently) Reads image path from command line.     #
#           (Ideally:)                                          #
#           Array of images(as numpy array of original size or  #
#           3*416*416)                                          #
#           OR                                                  #
#           List of paths of images in a batch                  #
#                                                               #
# Returns : Output of 58th convolutional layer of yolov3.       #
#           i.e. layer 'conv_80'                                #
#           (Currently) shape- 1 * 13 * 13 * 1024               #
#           (Ideally) shape- (batch_size) 13 * 13 * 1024.       #
#                                                               #
#################################################################

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
from DataLoader import *

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = RefDataset("train")
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=(8 if device == "cuda" else 0))


img_yolo_embedding = {}
for idx, (file_names,file_ids) in enumerate(loader):
	# print(idx,len(file_names))
	img_yolo_embedding = img_yolo_embedding.fromkeys(file_ids)
	imgs =[]

	for f in file_names:
		f = os.path.join('data/images/train2014',f)
		if not os.path.isfile(f):
			print("Input image file ", f, " doesn't exist")
			sys.exit(1)

		img = cv.imread(f)
		imgs.append(img)
	imgs = np.asarray(imgs)

	# Create a 4D blob from a frame.
	blob = cv.dnn.blobFromImages(imgs, 1/255, (Width, Height), [0,0,0], 1, crop=False)
	print (blob.shape)
	
	# Sets the input to the network
	net.setInput(blob)
	
	# Runs the forward pass to get output of the output layers
	# layersNames = net.getLayerNames()
	if idx == 0:
		out = net.forward('conv_80')
		print (out.shape)
	else:
		out = np.concatenate((out, net.forward('conv_80')), axis=0)
		print (out.shape)