############################################################
# NETWORK
# Instantiate Network Class that contains all modules

# FORWARD:
# Visual Input from Dataloader to CNN + Textual Input 
# from Dataloader to BiLSTM

# Concatenate CNN output + BiLSTM output -> Combined input

# Pass the combined input to the Attention module, 
# Feature Predictor and BBox CNN

# Obtain three outputs used by main file to compute losses 
# and train the model
############################################################
############
# PACKAGES #
############
import torch
import torch.nn as nn

###########
# MODULES #
###########
import ImageCNN
import TextBiLSTM
import AttentionModule
import FeaturePredictor
from BoundingBoxPredictor import BBP

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.cnn = None
        self.bilstm = None
        self.attn = None
        self.fp = None
        self.bbp = BBP()
        
        # Initialize self.CNN, self.BiLSTM, self.Attention, self.FP, self.BBP
        # based on classes defined in other modules

    def forward(self, visual_in, textual_in):

        # Call respective forward functions of each module
        # After each step, might need to call transformation functions on output
        # of previous module before passing it to the next

        # CNN: 
        #   Input Image
        #   - batch_size x 3 x 416 (height) x 416 (width)
        #   Outputs visual embedding
        #   - batch_size x 13 x 13 (# of local regions=169) x 1024 (length of visual embedding per region)

        # BiLSTM: 
        #   Input Text 
        #   - batch_size x (variable seq_len) 
        #   Outputs textual embedding 
        #   - batch_size x 2048 (4 (fwd + bckwd outputs for 2-layer BiLSTM) x 512)

        # Attention:
        #   Inputs: visual embedding + textual embedding
        #   - V: batch_size x 13 x 13 x 1024
        #   - T: batch_size x 2048
        #   Outputs combined embedding
        #   - batch_size x 1024 (x 1 x 1, might be changed to x 2 x 2 or something larger later on)

        # FP:
        #   Input: combined embedding
        #   - batch_size x 1024
        #   Outputs vector of probabilitistic predictions of each of the most frequent attributes in the dataset
        #   - batch_size x num_attr (in paper, num_attr = 50)

        # BBP:
        #   Input: combined embedding + textual embedding (concat within BBP forward for now)
        #   - C: batch_size x 1024
        #   - T: batch_size x 2048
        #   Outputs bbox prediction
        #   - batch_size x 5 (x, y, w, h, conf)