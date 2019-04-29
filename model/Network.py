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
from AttentionModule import ImageTextAttention
# import FeaturePredictor
from BoundingBoxPredictor import BBP

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.attn = ImageTextAttention()
        self.bbp = BBP()

        # TODO: Initialize following loss weight factors appropriately
        self.lweight = None
        self.cweight = None
        self.aweight = None

    def forward(self, img_feats, txt_feats):

        # img_feats: batch_size x 1024 x 13 x 13
        # txt_feats: batch_size x 2048

        agg_feats = self.attn(img_feats, txt_feats) # batch_size x 1024 x 1 x 1
        
        agg_feats = agg_feats.permute(0, 3, 1, 2)   # batch_size x 1 x 1 x 1024
        txt_feats = txt_feats.unsqueeze(1).unsqueeze(2) # batch_size x 1 x 1 x 2048
        preds = self.bbp(agg_feats, txt_feats) # batch_size x 5

        return preds

    def loss(self, bounding_boxes):
        # bounding_boxes: batch_size x 4
        lloss, closs = self.bbp.loss(bounding_boxes)
        aloss = self.attn.loss(bounding_boxes)

        loss = (self.lweight * lloss) + (self.cweight * closs) + (self.aweight * aloss)
        return loss

