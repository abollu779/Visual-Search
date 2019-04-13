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


class Network(nn.Module):
    def __init__(self,MODIFY_PAPER=False):

        ###################
        # SIZE DFINITIONS #
        ###################

        # Image features
        self.image_channels = 1024
        self.image_height = 32 if MODIFY_PAPER else 13
        self.image_width = 32 if MODIFY PAPER else 13
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


    def forward(self, image_features, text_features):

        # Check some shapes
        batch_size = image_features.shape[0]
        assert(image_features.shape == (batch_size,self.image_channels,self.image_height,self.image_width))
        assert(text_features.shape == (batch_size,self.text_feature_size))

        # Reshape the input
        image_features = image_features.permute(0,2,3,1)
        assert(image_features.shape == (batch_size,self.image_height,self.image_width,self.image_channels))
        image_features = image_features.view(batch_size,self.image_height*self.image_width,self.image_channels)
        assert(image_features.shape == (batch_size,self.num_image_feat_vecs,self.image_channels))

        # Project feature vectors to shared space
        proj_image_features = self.image_proj_matrix(image_features)
        assert(proj_image_features.shape == (batch_size,self.num_image_feat_vecs,self.hidden_dim_size))
        proj_text_features = self.text_proj_matrix(text_features)
        assert(proj_text_features.shape == (batch_size,self.hidden_dim_size))

        # Sum and apply activation
        combined_features = proj_image_features + proj_text_features
        assert(combined_features.shape == (batch_size,self.num_image_feat_vecs,self.hidden_dim_size))
        combined_features = self.alignment_activation(combined_features)
        assert(combined_features.shape == (batch_size,self.num_image_feat_vecs,self.hidden_dim_size))

        # Compute alignment coefficient for each local image feature
        # alignment matrix is really a collection of learnable alignment vectors, applied row-wise
        alignments = torch.stack([self.alignment_matrices[i](combined_features[:,i,:]) for i in range(self.num_image_feat_vecs)],dim=1)
        assert(alignments.shape == (batch_size,self.num_image_feat_vecs))

        # Weight each local image feature vector by its softmaxed alignment
        coeffs = self.softmax(alignments)
        assert(coeffs.shape == (batch_size,self.num_image_feat_vecs))
        weighted_feat_vecs = coeffs * image_features

        # Finally, combine the weighted feature vectors with some amount of summing
        aggregate_visual_feats = torch.zeros(batch_size,self.aggregate_feat_length,self.aggregate_image_height,self.aggregate_image_width)
        stride =
        for h in range(self.aggregate_image_height):
            for w in range(self.aggregate_image_width):
                aggregate_visual_feats[:,:,h,w] =

        # Cleanup for GPU's sake
        del image_features, text_features
        del proj_image_features, proj_text_features
        del combined_features
        del alignments, coeffs, weighted_feat_vecs

        return aggregate_visual_feats
