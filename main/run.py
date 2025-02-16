import torch
from torch.utils.data import DataLoader
import sys
import os

import config
import routine

sys.path.append(os.path.realpath('../model'))
from Network import Network, init_weights
from DataLoader import RefDataset, collate_fn

def run():
    print("Loading Data...")
    # Dataloading Section
    train_dataset = RefDataset("train")
    dev_dataset = None
    test_dataset = None

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    dev_loader = None
    test_loader = None

    print("Initializing Model...")
    # Network
    model = Network()
    model.apply(init_weights)
    model = model.to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)

    if config.train:
        print("=========Training Model=========")
        routine.routine(train_loader, dev_loader, model, optimizer)

    print("==========Inference===========")
    # Predict on Test Set
    # routine.predict(test_loader, model)
    return


if __name__=="__main__":
    run()



