import torch

x = torch.ones(5, requires_grad=True) # by default requires_grad is False, if True, it tell pytorch that it will need to calculate gradients later, 
print(x)