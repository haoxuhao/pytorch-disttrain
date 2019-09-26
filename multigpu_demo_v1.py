import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

# dataset
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

# model define
class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output

if __name__=="__main__":
    # Parameters
    input_size = 5
    output_size = 2

    batch_size = 30
    data_size = 100

    dataset = RandomDataset(input_size, data_size)
    # dataloader define
    rand_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size, shuffle=True)

    # model init
    model = Model(input_size, output_size)

    # cuda devices
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    for data in rand_loader:
        input = data.to(device)
        output = model(input)
        # loss

        # backward

        #update
        
        time.sleep(1)#simulate a long iteration time
        print("Outside: input size", input.size(),
            "output_size", output.size())

    torch.save(model.module.cpu().state_dict(), "model.pth")