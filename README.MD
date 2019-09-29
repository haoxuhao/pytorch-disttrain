# Introduction
This is a demo of pytorch distributed training. In this repo, you can find three simple demos for training model with several GPUs either on one single machine or several machines. The main code borrowed from [pytorch-multigpu](https://github.com/dnddnjs/pytorch-multigpu) and [pytorch-tutorial](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html). I only do some code finishing work, thanks to the two guy. What's more, a `sbatch` sample will be given for running distributed training on a HPC (High performance computer).

# Requirements
- Pytorch >= 1.0 is prefered.
- Python > 3.0 is preferd.
- NFS: all compute nodes are prefered to load data from the Network File System.
- linux: the pytorch distributed package can run on linux only now.
  
# Run the demos
## Demo 1 
This demo is based on the `torch.nn.DataParallel(model)`, the simplest one to use multi GPU on a single compute node. A batch is automatically divided into `N` mini-batches and processed by `N` GPUs. The models of different GPUs maintain synchronized during the whole training process.
``` shell
python multigpu_demo_v1.py
```
## Demo 2
This demo is based on the PyTorch distributed package. There exists `N` individual training processes and each process monopolizes a GPU. Also, the models on different GPUs maintain synchronized during the whole training process. We use `torch.distributed.launch` to create `N` processes.

``` shell
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 multigpu_demo_v2.py
```

## Demo 3
In this demo, I will run three processes on three different compute nodes, and each process can use either one GPU or several GPUs on that compute node (the same way as demo 1 sided) specified by `--gpu_devices 0 1`. Of course, every computes node must have the same PyTorch runtime environment. I use NFS file as the init_method. Note that the NFS file(not exists, maybe for UDS) must be accessed by all the processes on different compute nodes because all the processes need this file to communicate with each other during the initial cluster built process. The NFS file will be automatically removed after training. There are also other ways to init a process group, please refer to [here](https://tankzhou.cn/2019/07/07/Pytorch-%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83/).
### Manually launch a process on each computes node.
```
# node 1
python multigpu_demo_v3.py \
    --init_method file://<absolute path to nfs file> \
    --rank 0 \
    --world_size 3\
    --gpu_devices 0 1
# node 2
python multigpu_demo_v3.py \
    --init_method file://<absolute path to nfs file> \
    --rank 1 \
    --world_size 3\
    --gpu_devices 0 1
# node 3
python multigpu_demo_v3.py \
    --init_method file://<absolute path to nfs file> \
    --rank 2 \
    --world_size 3 \
    --gpu_devices 0 1
``` 
### GPU cluster on HPC

1. Create job.sh.
``` shell
#!/bin/bash

#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
#SBATCH --time=20:00
#SBATCH --output=log
#SBATCH --ntasks=1
#SBATCH --array=0,1

srun python multigpu_demo_v3.py \
    --init_method file://<absolute path to nfs file> \
    --rank $SLURM_ARRAY_TASK_ID \
    --world_size ${SLURM_ARRAY_TASK_COUNT}\
    --batch_size 256\
    --gpu_devices 0 1
```
2. Simply run `sbatch job.sh` on interactive node to submit a job.
# Performance comparisons
The code for the first three tests comes from [pytorch-multigpu](https://github.com/dnddnjs/pytorch-multigpu).
### settings
- gpu: p100;
- gpu cluster: 2 gpus/node;
- dataset: CIFAR10; batch size: 256; epoch: 1; iters: 196.
### results
| method | epoch time | batch time |
| --- | --- | --- |
| single gpu | 2:34 | 1.04s |
| v1 | 2:17 | 0.70s |
| v2 | 2:09 | 0.60s |
| v3 | 2:01 | 0.58s |

I didn't expect that the version(v2) of the two processes on a single machine would be slightly slower than the distributed version(v3) of the two nodes. Perhaps, it is due to the HPC. In short, no matter which way you use multiple GPUs, the speed will not increase by a multiple, because of the communication cost of the model synchronization. The only benefit you could get with multi-gpus is a bigger batch size.
<!--
batch size: 256, iters 196
single gpu: Training time 0:03:24.695632; Epoch: [180/196]| loss: 1.464 | acc: 45.362 | batch time: 1.041s
v1: 2 2gpu: Training time 0:02:17.989972; Epoch: [180/196]| loss: 2.040 | acc: 26.504 | batch time: 0.700s
v2: 2 gpu 2 process: Training time 0:02:09.304087; Epoch: [180/196]| loss: 1.443 | acc: 46.797 | batch time: 0.599s
v3: 2 gpu 2 nodes: Training time 0:02:01.753131; Epoch:[0][180/196]| loss: 1.447 | acc: 46.335 | batch time: 0.618s
-->

# Verify the models
This script will verify whether the models from different processes are synchronized.
```shell
python compare_models.py final_model_rank_0.pth final_model_rank_1.pth

# output
# layer3.15.bn3.running_var
# layer3.16.bn1.running_mean
# layer3.16.bn1.running_var
# layer3.16.bn2.running_mean
# layer3.16.bn2.running_var
# layer3.16.bn3.running_mean
# layer3.16.bn3.running_var
# bn_out.running_mean
# bn_out.running_var
```
From the ouput above, the only difference between models is **BN** layer, because that different minibatches' data does not synchronize. So this three way to facilitate multi gpus cannot improve the performance of the **BN**. If you want to improve your **BN** performance, the [**sync bn**](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) may satisfy your demand.

# Blogs
The chinese blogs:
- [Pytorch 多GPU训练-多计算节点并行-All you need](https://www.cnblogs.com/walter-xh/p/11613031.html)
- [Pytorch 多GPU训练-单运算节点-All you need](https://www.cnblogs.com/walter-xh/p/11586507.html)