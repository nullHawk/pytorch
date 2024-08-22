import torch
import numpy as np

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
print(type(b))

a.add_(1)
# when changing the value of a, b also changes, because we are using CPU and both are sharing the same memory
print(a)
print(b)

a = np.ones(5)
print(a)
b = torch.from_numpy(a) # by default datatype will be float64
print(b)

a += 1
# again when chaning the value of a, b also changes, because we are using CPU and both are sharing the same memory
print(a)
print(b) 