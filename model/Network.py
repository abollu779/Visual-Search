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
from TextBiLSTM import TextBiLSTM
from AttentionModule import ImageTextAttention
# import FeaturePredictor
from BoundingBoxPredictor import BBP

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.bilstm = TextBiLSTM()
        self.attn = ImageTextAttention()
        self.bbp = BBP()

        # Loss weight factors from paper
        # NOTE: Might need to review these weight factors, as we're missing
        # loss from attribute prediction phase at the moment
        self.lweight = 20.0
        self.cweight = 5.0
        self.aweight = 1.0

    def forward(self, img_feats, sent_feats):

        # img_feats: batch_size x 1024 x 13 x 13
        # sent_feats: batch_size x seq_len x 3072 (NOT A TENSOR)

        sent_feats = [torch.rand(5,3072),torch.rand(3,3072),torch.rand(2,3072), \
                      torch.rand(4,3072),torch.rand(2,3072),torch.rand(5,3072)]
        txt_feats = self.bilstm(sent_feats)
        # txt_feats: batch_size x 2048

        agg_feats = self.attn(img_feats, txt_feats) # batch_size x 1024 x 1 x 1
        
        agg_feats = agg_feats.permute(0, 2, 3, 1)   # batch_size x 1 x 1 x 1024
        txt_feats = txt_feats.unsqueeze(1).unsqueeze(2) # batch_size x 1 x 1 x 2048

        preds = self.bbp(agg_feats, txt_feats) # batch_size x 5

        return preds

    def loss(self, bounding_boxes):
        # bounding_boxes: batch_size x 4
        lloss, closs = self.bbp.loss(bounding_boxes)
        aloss = self.attn.loss(bounding_boxes)

        # Ignore Confidence Loss for Now
        loss = (self.lweight * lloss) + (self.aweight * aloss)
        return loss

def init_weights(m):
    if (type(m) == torch.nn.Linear) or (type(m) == torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
    elif type(m) == torch.nn.LSTM:
        for name, param in m.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)

