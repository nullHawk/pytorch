import torch

weights = torch.ones(4, requires_grad=True)

optimizer = torch.optim.SGD([weights], lr=0.01) # SGD is Stochastic Gradient Descent, lr is learning rate

optimizer.step() # update the weights
optimizer.zero_grad() # make the gradient zero after each epoch
