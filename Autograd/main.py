import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
print(y)

c = y*y*2
# c = c.mean()
print(c)

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
c.backward(v) # dc/dx, need to pass vector of same size as c, if c is scalar then no need to pass any argument
print(x.grad)


# x.requires_grad_(False) # stop tracking gradients
# x.detach() # stop tracking history of tensor
# with torch.no_grad(): # stop tracking history of tensor

