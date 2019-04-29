#################################################################
#                                                               #
# Saves :   Output of 58th convolutional layer of yolov3.       #
#           i.e. layer 'conv_80'                                #
#           shape- (num_images) * 1024 * 13 * 13                #
#                                                               #
#################################################################

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import pickle
import shelve
import time

from ImageLoader import *

Width  = 416       #Width of network's input image
Height = 416      #Height of network's input image
        
# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = ImageLoader()
loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=(8 if device == "cuda" else 0))

for idx, (img_names,img_IDs) in enumerate (loader):
	start = time.time()
	img_IDs = img_IDs.numpy()
	imgs =[]
	for f in img_names:
		f = os.path.join('data/images/train2014/',f)
		if not os.path.isfile(f):
			print("Input image file ", args.image, " doesn't exist")
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
	out = net.forward('conv_80')

	if os.path.isfile("ImageIDs.npy") and os.path.isfile("ImageEmbeddings.npy"):
		ids = np.load("ImageIDs.npy")
		ids = np.concatenate((ids,img_IDs))
		np.save("ImageIDs.npy",ids)

		embeddings = np.load("ImageEmbeddings.npy")
		embeddings = np.concatenate((embeddings,out))
		np.save("ImageEmbeddings.npy",embeddings)
	else:
		print (not( os.path.isfile("ImageIDs.npy") or os.path.isfile("ImageEmbeddings.npy")))
		assert(not( os.path.isfile("ImageIDs.npy") or os.path.isfile("ImageEmbeddings.npy")))
		np.save("ImageIDs.npy",img_IDs)
		np.save("ImageEmbeddings.npy",out)
	end = time.time()
	print ("Batch#",idx,"saved", "time =", end-start)