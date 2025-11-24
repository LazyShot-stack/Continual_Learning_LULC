import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):
        """
        Args:
            model: The PyTorch model.
            dataset: A list of data samples (or a DataLoader) to calculate Fisher Information.
        """
        self.model = model
        self.dataset = dataset
        
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._fisher = {}
        
        self.calculate_fisher()
        
    def calculate_fisher(self):
        """
        Calculates the Fisher Information Matrix for the current task.
        """
        self._fisher = {}
        self._means = {}
        
        # Store current parameters as the center of the gaussian
        for n, p in self.params.items():
            self._means[n] = p.clone().data
            self._fisher[n] = torch.zeros_like(p.data)

        self.model.eval()
        
        # Iterate over a subset of data to estimate Fisher Information
        # Assuming dataset is a DataLoader
        for i, batch in enumerate(self.dataset):
            inputs = batch['image']
            targets = batch['label']
            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate negative log-likelihood (CrossEntropy)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            
            for n, p in self.params.items():
                if p.grad is not None:
                    self._fisher[n] += p.grad.data ** 2
            
            # Limit to a few batches to save time if dataset is huge
            if i > 50: 
                break
        
        # Normalize
        for n in self._fisher:
            self._fisher[n] /= (i + 1)
            
    def penalty(self, model: nn.Module):
        """
        Calculates the EWC penalty loss.
        """
        loss = 0
        for n, p in model.named_parameters():
            if n in self._fisher:
                _loss = self._fisher[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss
