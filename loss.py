import torch 
import torch.nn as nn
import torch.nn.functional as F

def loss_function(predic, mask, weight=None):
	criterion = nn.NLLLoss(weight=weight)
	#criterion = nn.BCEWithLogitsLoss(weight=weight)
	predic = F.log_softmax(predic)
	loss = criterion(predic, mask)
	return loss