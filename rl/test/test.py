import numpy as np
import pandas as pd
import torch

a = pd.DataFrame(np.ones((3,4)))
a.ix[1, :] *= 0
a.ix[1, 1] = 1

print(a)

n_data = torch.ones(1, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
tensorx = torch.FloatTensor([[1,2]])
print(tensorx)


outs = [1, 2, 3]
tensor = torch.FloatTensor(outs)
sss = torch.stack(tensor, dim=1)
print(sss)