import torch
import time

import config

def routine(train_loader, dev_loader, model, optimizer):
    for epoch in range(config.num_epochs):
        before_epoch = time.time()
        train_loss = train_epoch(train_loader, model, optimizer)
        # val_loss = evaluate(dev_loader, model)
        
        after_epoch = time.time()
        epoch_time = after_epoch - before_epoch
    return

def train_epoch(train_loader, model, optimizer):
    model.train()
    epoch_loss = 0
    num_batches = len(train_loader)
    
    for batch_id, (img_feats, txt_feats, bboxes) in enumerate(train_loader):
        optimizer.zero_grad()
        img_feats = img_feats.to(config.device)
        txt_feats = txt_feats.to(config.device)
        bboxes = bboxes.to(config.device)
        
        model(img_feats, txt_feats)

        # Compute loss
        loss = model.loss(bboxes)
        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()
        print("Batch: %d | Loss: %.6f" % (batch_id, loss.item()))

    print("Avg Epoch Loss: %.6f" % (epoch_loss/num_batches))

    return epoch_loss/num_batches

def evaluate(dev_loader, model):
    model.eval()
    val_loss = 0
    num_batches = len(dev_loader)

    for batch_id, (img_feats, txt_feats, bboxes) in enumerate(dev_loader):
        img_feats = img_feats.to(config.device)
        txt_feats = txt_feats.to(config.device)
        
        model(img_feats, txt_feats)

        # Compute loss
        loss = model.loss(bboxes)
        val_loss += loss.item()

    return val_loss/num_batches

def predict(test_loader, model):
    model.eval()

    for batch_id, (img_feats, txt_feats) in enumerate(test_loader):
        img_feats = img_feats.to(config.device)
        txt_feats = txt_feats.to(config.device)

        preds = model(img_feats, txt_feats)
        
        # Write Code Here To Visualize Predictions

    return
