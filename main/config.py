import torch

#############
## GENERAL ##
#############
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train = True
batch_size = 3
num_workers = 8 if (device=='cuda') else 0

#######################
# Network Hyperparams #
#######################

learning_rate = 1e-3
# weight_decay = 1e-6
momentum=0.9
num_epochs = 3