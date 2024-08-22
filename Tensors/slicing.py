import torch

x = torch.rand(5, 3)

print(x)
print(x[:, 0]) # print all rows of 0th column
print(x[1,1]) # print value at 1st row and 1st column
print(x[1,1].item()) # get value of tensor as python number


# view() is used to reshape the tensor
x = torch.randn(4, 4)
print(x.view(-1, 8))