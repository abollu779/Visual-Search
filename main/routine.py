import torch
import time

import config

def routine(train_loader, dev_loader, model, optimizer, criterion):
    for epoch in range(config.num_epochs):
        before_epoch = time.time()
        train_loss = train_epoch(train_loader, model, optimizer, criterion)
        val_loss = evaluate(dev_loader, model, criterion)
        
        after_epoch = time.time()
        epoch_time = after_epoch - before_epoch
    return

def train_epoch(train_loader, model, optimizer, criterion):
    model.train()
    epoch_loss = 0
    num_batches = len(train_loader)
    
    for batch_id, (img_feats, txt_feats, bboxes) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.to(config.device)
        descriptions = descriptions.to(config.device)
        
        pred_bboxes = model(img_feats, txt_feats)

        # Compute loss
        loss = None
        # EXAMPLE: loss = criterion(pred_bboxes, bboxes)
        # ^The call depends on how we create our Loss class, what it computes
        # and what inputs it needs as input

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss/num_batches

def evaluate(dev_loader, model, criterion):
    model.eval()
    val_loss = 0
    num_batches = len(dev_loader)

    for batch_id, (img_feats, txt_feats, bboxes) in enumerate(dev_loader):
        images = images.to(config.device)
        descriptions = descriptions.to(config.device)
        
        pred_bboxes = model(img_feats, txt_feats)

        # Compute loss
        loss = None
        # EXAMPLE: loss = criterion(pred_bboxes, bboxes)
        # ^The call depends on how we create our Loss class, what it computes
        # and what inputs it needs as input

        val_loss += loss.item()

    return val_loss/num_batches

