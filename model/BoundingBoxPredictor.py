############################################################
# BOUNDING BOX PREDICTOR
# Instantiate BBP Class
# Input: Single 3072 dimensional vector
# Output: Sigmoid output along 5 features (x, y, width, height, conf)

# Should have CNNs (1,1: 3072 filters and 1,1: 5 filters)
# Each convolution followed by Leaky ReLU
# After convolutions, pass through a sigmoid 
############################################################

import torch
import torch.nn as nn

class BPP(nn.Module):
    def __init__(self):
        conv1 = nn.Conv2d(1, 3072, 1)
        lrelu1 = nn.LeakyReLU()

        conv2 = nn.Conv2d(3072, 5, 1)
        lrelu2 = nn.LeakyReLU()

        sigmoid = nn.Sigmoid()

        self.layers = nn.Sequential(conv1, lrelu1, conv2, lrelu2, sigmoid)

    def forward(self, combined_visual_feat, textual_feat):
        # textual_feat: batch_size x 2048
        # combined_visual_feat: batch_size x 1024
        concat_input = torch.cat((combined_visual_feat, textual_feat), 1)
        bbox_pred = self.layers(concat_input)
        return bbox_pred