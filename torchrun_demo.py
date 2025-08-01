import os
import sys
import socket

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from datautils import MyTrainDataset
from loguru import logger

def get_dist_info():
    env={
        'world_size': int(os.environ["WORLD_SIZE"]),
        'global_rank': int(os.environ["RANK"]),
        'local_rank': int(os.environ["LOCAL_RANK"]),
        'master_addr': os.environ["MASTER_ADDR"],
        'master_port': os.environ["MASTER_PORT"],
        'hostname': socket.gethostname()
    }
    return env

def ddp_setup():
    host_info = get_dist_info()
    host_string = "[{} GR {} LR {}]".format(
        host_info["hostname"],
        host_info["global_rank"],
        host_info["local_rank"],
    )

    logger.remove()
    logger.add(
        sink=sys.stderr,  # or a file path like "log.txt"
        format= host_string + " | {time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
    )

    logger.info("Setting up DDP " + " world_size: {}  master_addr: {}  master_port: {}".format(
        host_info["world_size"],
        host_info["master_addr"],
        host_info["master_port"]
    ))
    init_process_group(backend="nccl")
    logger.info("DDP initialized")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        env: dict
    ) -> None:
        self.env=env
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            logger.info("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        logger.info(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        logger.log("INFO","Epoch {} | Batchsize: {} | Steps: {}".format(epoch,b_sz,len(self.train_data)))
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        logger.info(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(args):
    ddp_setup()
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, args.batch_size)
    env = get_dist_info()
    trainer = Trainer(model, train_data, optimizer, args.save_every, args.snapshot_path,env)
    trainer.train(args.total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', type=int, help='Total epochs to train the model', required = True)
    parser.add_argument('--save_every', type=int, help='How often to save a snapshot', required = True)
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--snapshot_path',type=str,default="snapshot.pt")
    args = parser.parse_args()
    
    main(args)