import torch
labels = torch.tensor([1,0,1,2,3,4,5,9])
labels = labels.repeat_interleave(10)
print(labels)