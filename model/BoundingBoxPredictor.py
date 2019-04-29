############################################################
# BOUNDING BOX PREDICTOR
# Instantiate BBP Class
# Input:
# 1. Aggregated Visual Feature: N x 1 x 1 x 1024
# 2. Textual Feature: N x 1 x 1 x 2048

# Output: 
# 1. Sigmoid computed along the 5 features (x, y, w, h, conf): N x 1 x 1 x 5
# Sigmoid output along 5 features (x, y, width, height, conf)

# Should have CNNs (1,1: 3072 filters and 1,1: 5 filters)
# Each convolution followed by Leaky ReLU
# After convolutions, pass through a sigmoid 
############################################################

import torch
import torch.nn as nn

class BBP(nn.Module):
    def __init__(self):
        super(BBP, self).__init__()
        self.predictions = None

        ##########
        # CONFIG #
        ###########
        self.input_img_height = 416
        self.input_img_width = 416
        self.iou_thresh = 0.5

        #######################
        # MODULE ARCHITECTURE #
        #######################
        # First Convolution Layer:
        conv1 = nn.Conv2d(3072, 3072, 1)
        lrelu1 = nn.LeakyReLU()
        # Second Convolution Layer:
        conv2 = nn.Conv2d(3072, 5, 1)
        lrelu2 = nn.LeakyReLU()
        # Sigmoid Prediction Layer:
        sigmoid = nn.Sigmoid()

        self.layers = nn.Sequential(conv1, lrelu1, conv2, lrelu2, sigmoid)

    def forward(self, combined_visual_feat, textual_feat):
        ###################
        # CONCAT FEATURES #
        ###################
        # combined_visual_feat: N x 1 x 1 x 1024
        # textual_feat: N x 1 x 1 x 2048
        concat_input = torch.cat((combined_visual_feat, textual_feat), 3)
        concat_input = concat_input.permute(0, 3, 1, 2)
        # concat_input: N x 3072 x 1 x 1

        ################
        # PREDICT BBOX #
        ################
        bbox_pred = self.layers(concat_input)

        self.predictions = bbox_pred
        # self.predictions: N x 5 x 1 x 1 (tx, ty, tw, th, tc)

        # Computing Bounding Box Coordinates:
        # bx = tx * pw
        # by = ty * ph
        # bw = tw^2 * pw
        # bh = th^2 * ph
        return self.predictions

    def localization_loss(self, preds, ground_truths):
        #####################
        # COORDINATE LOSSES #
        #####################
        tx, ty, tw, th = preds.permute(1,0)
        gt_bx1, gt_by1, gt_bx2, gt_by2 = ground_truths.permute(1,0)

        gt_bw = gt_bx2 - gt_bx1 + 1
        gt_bh = gt_by2 - gt_by1 + 1

        lx = (tx - (gt_bx1/self.input_img_width)) ** 2
        ly = (ty - (gt_by1/self.input_img_height)) ** 2

        ####################
        # DIMENSION LOSSES #
        ####################
        lw = (tw - torch.sqrt(gt_bw/self.input_img_width)) ** 2
        lh = (th - torch.sqrt(gt_bh/self.input_img_height)) ** 2

        loss =  lx + ly + lw + lh
        return loss.mean()

    def compute_iou(self, preds, ground_truths):
        # preds: N x 5
        batch_size = preds.size()[0]
        tx, ty, tw, th = preds.permute(1,0)
        gt_bx1, gt_by1, gt_bx2, gt_by2 = ground_truths.permute(1,0)
        gt_w = gt_bx2 - gt_bx1 + 1
        gt_h = gt_by2 - gt_by1 + 1

        x_l = torch.max(tx, gt_bx1)
        x_r = torch.min(tx+tw-1, gt_bx2)
        y_t = torch.max(ty, gt_by1)
        y_b = torch.min(ty+th-1, gt_by2)

        zeros = torch.zeros(batch_size)
        intersection = torch.max(zeros, x_r-x_l+1) * torch.max(zeros, y_b-y_t+1)
        union = (tw*th) + (gt_w * gt_h) - intersection

        iou = intersection / union
        return iou

    def conf_loss(self, preds, ground_truths):
        iou = self.compute_iou(preds[:,:-1], ground_truths)

        tc = preds[:,-1]

        gt_bc = iou >= self.iou_thresh

        loss = (gt_bc.float() * torch.log(tc)) + ((1 - gt_bc.float()) * torch.log(1-tc))

        return loss.mean()

    def loss(self, bounding_boxes):
        preds = torch.squeeze(self.predictions)
        # bounding_boxes: batch_size x 4
        # preds: batch_size x 5

        lloss = self.localization_loss(preds[:,:-1], bounding_boxes)
        closs = self.conf_loss(preds, bounding_boxes)

        return lloss, closs


def test_module():
    print("=========BEGIN TESTING=========")
    # Initialize Module
    bbp = BBP()
    optim = torch.optim.SGD(bbp.parameters(), lr = 0.01, momentum=0.9)

    # Initialize Random Initial Input Tensors
    agg_feats = torch.rand(2, 1, 1, 1024)
    textual_feats = torch.rand(2, 1, 1, 2048)
    bounding_boxes = torch.rand(2, 4)

    preds = bbp.forward(agg_feats, textual_feats)
    print("Predictions Shape: ", preds.size())

    lloss, closs = bbp.loss(bounding_boxes)

    # TODO: total_loss needs to be a weighted sum of 4 individual losses
    total_loss = lloss + closs

    total_loss.backward()
    optim.step()
    print("=========END TESTING=========")
    




if __name__ == "__main__":
    test_module()



