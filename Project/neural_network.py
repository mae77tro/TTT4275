import knn as knn

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class NeuralNetwork(nn.Module):
	def __init__(self,in_features=62,h1=32,h2=16,out_features=10):
		super().__init__()
		
        # define layers
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			# We have 62 features as inputs
			nn.Linear(62, 32),
			nn.ReLU(),
			nn.Linear(32, 16),
			nn.ReLU(),
			nn.Linear(16, 10),
		)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits





device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)

model = NeuralNetwork().to(device)
print(model)