import torch

#############
## GENERAL ##
#############
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train = True

#######################
# Network Hyperparams #
#######################

# TODO: Change these values accordingly when we start training #
learning_rate = 1e-3
weight_decay = 1e-6
num_epochs = 3