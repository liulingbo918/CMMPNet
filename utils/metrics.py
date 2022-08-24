import torch
import torch.nn as nn
import cv2
import numpy as np


class IoU(nn.Module):
    def __init__(self, threshold=0.5):
        super(IoU, self).__init__()
        self.threshold = threshold

    def resize(self,x, h, w):
        b = x.shape[0]
        x = x.cpu().detach().numpy() 
        y = np.zeros((b, x.shape[1], h, w))
        for id in range(b):
            x1 = x[id,:,:,:].transpose(1,2,0)
            y[id,:,:,:] = cv2.resize(x1, (h, w))
        return torch.Tensor(y)

    def forward(self, target, inputs):
        # the inputs should be resized to the resolution of the target for evaluation
        if inputs.shape[2] != target.shape[2] or inputs.shape[3] != target.shape[3]:
           inputs = self.resize(inputs, target.shape[2], target.shape[3]).cuda()
        
        eps = 1e-10
        input_  = (inputs > self.threshold).data.float()
        target_ = (target > self.threshold).data.float()
        
        intersection = torch.clamp(input_ * target_, 0, 1)  
        union        = torch.clamp(input_ + target_, 0, 1)

        ## batch_iou is a single value
        batch_iou = 0.0 if torch.mean(intersection).lt(eps) else float(torch.mean(intersection).cpu().data / torch.mean(union).cpu().data) 

        ## inter_pixel_num and union_pixel_num are two vectors with [batch_size] values. 
        samples_inter_pixel_num = torch.sum(intersection, (1,2,3))
        samples_union_pixel_num = torch.sum(union,        (1,2,3))
        
        return batch_iou, samples_inter_pixel_num, samples_union_pixel_num

