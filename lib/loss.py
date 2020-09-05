import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative, size_average = True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        
        if size_average:
            losses = losses.mean()
        else:
            losses = losses.sum()
            
        return losses
   
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)
        losses = (target.float()*distances +
                 (1 + -1*target).float() * F.relu(self.margin-distances.sqrt()).pow(2))
        
        if size_average:
            losses = losses.mean()
        else:
            losses = losses.sum()
        
        return losses
