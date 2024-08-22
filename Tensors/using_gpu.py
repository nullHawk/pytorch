import torch

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda") # a CUDA device object


x = torch.ones(5, device=device) # directly create a tensor on GPU if it is present
# x.to(device) # move tensor manually to different device

# remember that you cannot use GPU tensor in numpy, it handles only cpu tensors, you need to move back tensor to CPU to use it in numpy