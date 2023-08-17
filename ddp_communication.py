import os
import argparse
import socket

import torch
import torch.distributed as dist

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])
hostname=socket.gethostname()

def run(backend):
    tensor = torch.zeros(1)
    
    # Need to put tensor on a GPU device for nccl backend
    if backend == 'nccl':
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        tensor = tensor.to(device)

    if WORLD_RANK == 0:
        for rank_recv in range(1, WORLD_SIZE):
            dist.send(tensor=tensor, dst=rank_recv)
            print('[{}] worker_{} sent data to Rank {} world_size {}\n'.format(hostname,0, rank_recv,WORLD_SIZE))
    else:
        dist.recv(tensor=tensor, src=0)
        print('[{}] worker_{} (local_rank {}) has received data from rank {}\n'.format(hostname,WORLD_RANK,LOCAL_RANK, 0))

def init_processes(backend):
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    run(backend)
    dist.destroy_process_group()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="nccl", choices=['nccl', 'gloo'])
    args = parser.parse_args()

    init_processes(backend=args.backend)