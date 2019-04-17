######################################################
#                                                    #
# Attention Module                                   #
#                                                    #
# Inputs: Image Feature Vector from CNN SI           #
#         Textual Feature Vector from LSTM VE        #
#                                                    #
# Outputs: Resultant "aggregated" visual features VI #
#                                                    #
# VI = SUM( Softmax( alignment( SI, VE ) ) * SI)     #
#                                                    #
######################################################


if __name__ == '__main__':
    import torch
    from torch import nn


class ImageTextAttention(nn.Module):
    def __init__(self,MODIFY_PAPER=False):
        super(ImageTextAttention, self).__init__()

        ###################
        # SIZE DFINITIONS #
        ###################

        # Image features
        self.image_channels = 1024
        self.image_height = 32 if MODIFY_PAPER else 13
        self.image_width = 32 if MODIFY_PAPER else 13
        self.num_image_feat_vecs = self.image_height * self.image_width

        # Text features
        self.text_feature_size = 2048

        # Shared space into which text and image features can be projected
        self.hidden_dim_size = 512

        # Output aggregate image features
        self.aggregate_image_height = 4 if MODIFY_PAPER else 1
        self.aggregate_image_width = 4 if MODIFY_PAPER else 1
        self.aggregate_feat_length = self.image_channels

        ############################
        # PARAMATERS AND FUNCTIONS #
        ############################

        self.coeffs = None

        # This is the notation used in the paper
        N = self.num_image_feat_vecs
        H = self.hidden_dim_size

        # Three matrices to be learned
        self.image_proj_matrix = nn.Linear(self.image_channels,H,bias=False)
        self.text_proj_matrix = nn.Linear(self.text_feature_size,H,bias=False)
        self.alignment_matrices = [nn.Linear(H,1,bias=False) for _ in range(N)]

        # Activation functions
        self.alignment_activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)

        # Distance metric for loss function
        self.dist = nn.PairwiseDistance()


    def forward(self, image_features, text_features):

        # Check some shapes
        batch_size = image_features.shape[0]
        assert(image_features.shape == (batch_size,self.image_channels,self.image_height,self.image_width))
        assert(text_features.shape == (batch_size,self.text_feature_size))

        # Reshape the input
        image_features = image_features.permute(0,2,3,1)
        assert(image_features.shape == (batch_size,self.image_height,self.image_width,self.image_channels))
        image_features = image_features.view(batch_size,-1,self.image_channels)
        assert(image_features.shape == (batch_size,self.num_image_feat_vecs,self.image_channels))

        # Project feature vectors to shared space
        proj_image_features = self.image_proj_matrix(image_features)
        assert(proj_image_features.shape == (batch_size,self.num_image_feat_vecs,self.hidden_dim_size))
        proj_text_features = self.text_proj_matrix(text_features)
        assert(proj_text_features.shape == (batch_size,self.hidden_dim_size))
        proj_text_features = proj_text_features.unsqueeze(dim=1)
        assert(proj_text_features.shape == (batch_size,1,self.hidden_dim_size))

        # Sum and apply activation
        combined_features = proj_image_features + proj_text_features
        assert(combined_features.shape == (batch_size,self.num_image_feat_vecs,self.hidden_dim_size))
        combined_features = self.alignment_activation(combined_features)
        assert(combined_features.shape == (batch_size,self.num_image_feat_vecs,self.hidden_dim_size))

        # Compute alignment coefficient for each local image feature
        # alignment matrix is really a collection of learnable alignment vectors, applied row-wise
        alignments = torch.stack([self.alignment_matrices[i](combined_features[:,i,:]) for i in range(self.num_image_feat_vecs)],dim=1)
        assert(alignments.shape == (batch_size,self.num_image_feat_vecs,1))
        alignments = alignments.squeeze()
        assert(alignments.shape == (batch_size,self.num_image_feat_vecs))

        # Weight each local image feature vector by its softmaxed alignment
        self.coeffs = self.softmax(alignments)
        assert(self.coeffs.shape == (batch_size,self.num_image_feat_vecs))
        self.coeffs = self.coeffs.unsqueeze(dim=2)
        assert(self.coeffs.shape == (batch_size,self.num_image_feat_vecs,1))
        weighted_feat_vecs = self.coeffs * image_features
        assert(weighted_feat_vecs.shape == (batch_size,self.num_image_feat_vecs,self.image_channels))
        weighted_feat_vecs = weighted_feat_vecs.view(batch_size,self.image_height,self.image_width,self.image_channels)
        assert(weighted_feat_vecs.shape == (batch_size,self.image_height,self.image_width,self.image_channels))

        # Finally, combine the weighted feature vectors with some amount of summing
        aggregate_visual_feats = torch.zeros(batch_size,self.aggregate_feat_length,self.aggregate_image_height,self.aggregate_image_width)
        stride = self.image_width // self.aggregate_image_width
        for h in range(self.aggregate_image_height):
            for w in range(self.aggregate_image_width):
                aggregate_visual_feats[:,:,h,w] = torch.sum(weighted_feat_vecs[:,h*stride:(h+1)*stride,w*stride:(w+1)*stride,:],dim=(1,2))
        assert(aggregate_visual_feats.shape == (batch_size,self.aggregate_feat_length,self.aggregate_image_height,self.aggregate_image_width))

        # Cleanup for GPU's sake
        del image_features, text_features
        del proj_image_features, proj_text_features
        del combined_features
        del alignments, weighted_feat_vecs

        return aggregate_visual_feats


    def loss(self,bounding_boxes):

        batch_size = bounding_boxes.shape[0]
        truth = torch.zeros(batch_size,self.image_height,self.image_width)

        for i in range(batch_size):
            for h in range(self.image_height):
                for w in range(self.image_width):

                    # Parse bounding box
                    x1,y1,width,height = bounding_boxes[i]
                    x2,y2 = x1+width,y1+height

                    # Convert box coords from [0,1] to image resolution
                    x1 *= self.image_width
                    x2 *= self.image_width
                    y1 *= self.image_height
                    y2 *= self.image_height

                    # Determine value that alignment should have been
                    coverage = (min(w+1,x2)-max(w,x1))*(min(h+1,y2)-max(h,y1))
                    coverage = min(coverage,1)
                    coverage = max(coverage,0)
                    truth[i,h,w] = coverage

        # Calculate the loss
        assert(self.coeffs.shape == (batch_size,self.num_image_feat_vecs,1))
        truth = truth.view(batch_size,-1,1)
        assert(truth.shape == (batch_size,self.num_image_feat_vecs,1))
        loss = self.dist(self.coeffs,truth).mean()
        return loss


if __name__ == '__main__':

    def test_attention():

        dut = ImageTextAttention(MODIFY_PAPER=False)
        optimizer = torch.optim.Adam(dut.parameters())

        image = torch.rand(2,1024,13,13)
        text = torch.rand(2,2048)
        true_bounding_boxes = torch.rand(2,4)

        out = dut.forward(image,text)
        assert(out.shape == (2,1024,1,1))
        loss = dut.loss(true_bounding_boxes)
        loss.backward()
        optimizer.step()

        print("Test did not crash with MODIFY_PAPER=False")

        dut = ImageTextAttention(MODIFY_PAPER=True)
        loss_fn = nn.PairwiseDistance()
        optimizer = torch.optim.Adam(dut.parameters())

        image = torch.rand(2,1024,32,32)
        text = torch.rand(2,2048)
        true_bounding_boxes = torch.rand(2,4)

        out = dut.forward(image,text)
        assert(out.shape == (2,1024,4,4))
        loss = dut.loss(true_bounding_boxes)
        loss.backward()
        optimizer.step()

        print("Test did not crash with MODIFY_PAPER=True")

    test_attention()
    print("SUCCESS")
