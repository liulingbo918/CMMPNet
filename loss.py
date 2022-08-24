import torch
import torch.nn as nn
import cv2
import numpy as np


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1.0 
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def resize(self, y_true, h, w):
        b = y_true.shape[0]
        y = np.zeros((b, y_true.shape[1], h, w))
        
        y_true = np.array(y_true.cpu())
        for id in range(b):
            y1 = y_true[id,:,:,:].transpose(1,2,0)
            y[id,:,:,:] = cv2.resize(y1, (h, w))
        return torch.Tensor(y)
        
    def __call__(self, y_true, y_pred):
        # the ground_truth map is resized to the resolution of the predicted map during training
        if y_true.shape[2] != y_pred.shape[2] or y_true.shape[3] != y_pred.shape[3]:
           y_true = self.resize(y_true, y_pred.shape[2], y_pred.shape[3]).cuda()
           
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b