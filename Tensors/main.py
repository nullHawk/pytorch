import torch

x = torch.empty(2, 2) # Initialized 2x2 tensor
y = torch.rand(2, 2) # Initialized 2x2 tensor with random values
zero = torch.zeros(2, 2) # Initialized 2x2 tensor with zeros
ones = torch.ones(2, 2) # Initialized 2x2 tensor with ones
# print(x)
# print(y)
# print(zero)
# print(ones.dtype) #check data type of tensor

ones = torch.ones(2, 2, dtype=torch.int) # Initialized 2x2 tensor with ones and data type as int
# print(ones.dtype) #check data type of tensor

# print(ones.size()) #check size of tensor

custom_tensor = torch.tensor([2.5, 0.1]) # Initialized tensor with custom values
# print(custom_tensor)

x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(x)
print(y)
z = x + y # or z = torch.add(x, y)
print(z)
print(y.add_(x)) # inplace addition    | In pytorch every function with _ is inplace function

print(torch.sub(x, y)) # subtraction
print(torch.mul(x, y)) # multiplication

