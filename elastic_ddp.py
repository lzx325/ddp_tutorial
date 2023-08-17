import os
import time
import logging
import socket
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

hostname=socket.gethostname()
global_rank = os.environ["RANK"]
local_rank = int(os.environ["LOCAL_RANK"])
log_format = f'{hostname}_{global_rank}_{local_rank}:%(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)

def ddp_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel,self).__init__()
        self.net1=nn.Linear(10,10)
        self.relu=nn.ReLU()
        self.net2=nn.Linear(10,5)
    def forward(self,x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic():
    ddp_setup()
    global_rank = os.environ["RANK"]
    local_rank = int(os.environ["LOCAL_RANK"])

    print(
        "hostname:",
        socket.gethostname(),
        "world size:",
        dist.get_world_size(),
        "global_rank:",
        global_rank,
        "local rank:",
        local_rank
    )

    device_id=local_rank % torch.cuda.device_count()
    model=ToyModel().to(device_id)
    ddp_model=DDP(model,device_ids=[device_id])
    loss_fn=nn.MSELoss()
    optimizer=optim.SGD(model.parameters(),lr=0.001)

    optimizer.zero_grad()
    outputs=ddp_model(torch.randn(20,10))
    print("after forward step")
    labels=torch.randn(20,5).to(device_id)
    loss_fn(outputs,labels).backward()
    print("after backward step")
    optimizer.step()
    print("after optimizer step")
    time.sleep(10)
    dist.destroy_process_group()

if __name__=="__main__":
    demo_basic()