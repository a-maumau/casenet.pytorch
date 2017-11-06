import torch 
import torch.nn as nn
import torch.nn.functional as F

def loss_function(output, mask):
	criterion = nn.NLLLoss2d()
	outputs = F.log_softmax(outputs)
    loss = criterion(outputs, mask)
    return loss