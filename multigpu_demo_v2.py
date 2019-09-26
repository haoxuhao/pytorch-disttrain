import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import torch.distributed as dist
import torch.utils.data.distributed
import sys
import time


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
        # print("\tIn Model: input size", input.size(),
        #       "output size", output.size())

        return output

if __name__=="__main__":
    # Parameters
    input_size = 5
    output_size = 2

    batch_size = 30
    data_size = 100

    # check the nccl backend
    if not dist.is_nccl_available():
        print("Error: nccl backend not available.")
        sys.exit(1)

    # init group
    dist.init_process_group(backend="nccl", init_method="env://")

    # get the process rank and the world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # prepare the dataset
    dataset = RandomDataset(input_size, data_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    

    rand_loader = DataLoader(dataset, batch_size=batch_size//world_size, 
                              shuffle=(train_sampler is None), 
                              sampler=train_sampler)

    # dataloader define
    # rand_loader = DataLoader(dataset=dataset,
    #                         batch_size=batch_size, shuffle=True)

    # model init
    model = Model(input_size, output_size)

    # cuda devices
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model)
    # model.to(device)

    # distribute model define
    device = torch.device('cuda', rank)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    print("From rank %d: start training, time:%s"%(rank, time.strftime("%Y-%m-%d %H:%M:%S")))
    for data in rand_loader:
        input = data.to(device)
        output = model(input)
        # loss

        # backward

        #update
        
        time.sleep(1)#模拟一个比较长的batch时间
        print("From rank %d: Outside: input size %s, output size %s"%(rank, str(input.size()), str(output.size())),flush=True)
    torch.save(model.module.cpu().state_dict(), "model_%d.pth"%rank)
    print("From rank %d: end training, time: %s"%(rank, time.strftime("%Y-%m-%d %H:%M:%S")))