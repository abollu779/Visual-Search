import torch
from torch.utils.data import DataLoader
import sys
import os

import config
import routine

sys.path.append(os.path.realpath('../model'))
from Network import Network
from DataLoader import RefDataset

def run():
    print("Loading Data...")
    # Dataloading Section
    train_dataset = RefDataset("train")
    dev_dataset = None
    test_dataset = None

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    dev_loader = None
    test_loader = None
    # EXAMPLE: train_loader, dev_loader, test_loader = load_data() where
    # load_data needs to be implemented to create appropriate dataloaders

    print("Initializing Model...")
    # Network
    model = Network()
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    if config.train:
        print("=========Training Model=========")
        routine.routine(train_loader, dev_loader, model, optimizer)

    print("==========Inference===========")
    # Predict on Test Set
    # routine.predict(test_loader, model)
    return


if __name__=="__main__":
    run()



