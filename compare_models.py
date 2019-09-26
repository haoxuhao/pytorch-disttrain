import torch
import sys

model1 = sys.argv[1]
model2 = sys.argv[2]

param1 = torch.load(model1, map_location="cpu")
param2 = torch.load(model2, map_location="cpu")

for key, v in param1.items():
    equal_flag = (torch.equal(v, param2[key]))
    if equal_flag == False:
        print(key)

