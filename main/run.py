import torch
import sys
import os

import config
import routine

sys.path.append(os.path.realpath('../model'))
from Network import Network

def run():
    # Dataloading Section
    train_loader, dev_loader, test_loader = None, None, None
    # EXAMPLE: train_loader, dev_loader, test_loader = load_data() where
    # load_data needs to be implemented to create appropriate dataloaders

    # Network
    model = Network()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Initialize criterion instance
    criterion = None
    # EXAMPLE: criterion = Loss(), where Loss is a class we define separately
    # which combines the four individual losses (confidence, localization, etc.)

    if config.train:
        print("=========Training Model=========")
        routine.routine(train_loader, dev_loader, model, optimizer, criterion)

    print("==========Inference===========")
    # Predict on Test Set
    # EXAMPLE: routine.predict(test_loader, model)
    return


if __name__=="__main__":
    run()



