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
        This implementation is robust to empty or very small datasets and
        will gracefully fall back to a zero Fisher matrix if calculation
        fails or no batches are available.
        """
        self._fisher = {}
        self._means = {}
        
        # Store current parameters as the center of the gaussian
        for n, p in self.params.items():
            self._means[n] = p.clone().data
            self._fisher[n] = torch.zeros_like(p.data)

        self.model.eval()
        
        num_batches = 0
        try:
            # Iterate over a subset of data to estimate Fisher Information
            # Assuming dataset is a DataLoader
            for batch in self.dataset:
                inputs = batch.get('image', None) if isinstance(batch, dict) else batch['image']
                targets = batch.get('label', None) if isinstance(batch, dict) else batch['label']

                if inputs is None or targets is None:
                    # Skip malformed batch
                    continue

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

                num_batches += 1
                # Limit to a few batches to save time if dataset is huge
                if num_batches > 50:
                    break
        except Exception as e:
            print(f"Warning: Fisher calculation failed: {e}. Using zero Fisher matrix.")
            # Leave _fisher as zeros and return
            return

        # Normalize if we processed any batches
        if num_batches > 0:
            for n in self._fisher:
                self._fisher[n] /= float(num_batches)
        else:
            print("Warning: No batches found for Fisher calculation; Fisher remains zero.")
            
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
