import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch_optimizer import RAdam
from torch_optimizer import AdamP
from tqdm import tqdm
from datetime import datetime
import os
import numpy as np
#import wandb

import hydra
from hydra.core.config_store import ConfigStore 
#from omegaconf import DictConfig

import aimat.core as am
import aimat.models as models
from config import AIMatConfig
from aimat.logger.tensorboard import TensorboardExperiment, WandBExperiment
from aimat.core.runner import Runner,run_epoch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank,world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

cs = ConfigStore.instance()
cs.store(name="AIMat_config", node=AIMatConfig)

@hydra.main(config_path="conf", config_name="config")
def main_worker(cfg: AIMatConfig):
    
    #check device mps or cuda or cpu 
    if "cuda" in cfg.params.device and torch.cuda. is_available():
        device = "cuda"
    elif "mps" in cfg.params.device and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}")
    
    #create dataloader

    train_loader, val_loader = am.set_dataloader(cfg.data.dataset_name,cfg.paths.data_dir, cfg.data.batch_size, cfg.data.batch_size_test, iscustom=cfg.data.iscustom)
   
    criterion = nn.__dict__[cfg.params.criterion]()
    model = models.__dict__[cfg.models.model_name]().to(device)
    learning_rate=cfg.params.learning_rate
    optimizer =optim.__dict__[cfg.params.optimizer](model.parameters(), lr=learning_rate)
    
    #if cfg.params.scheduler is not None: 
    #    scheduler = am.__dict__[cfg.params.scheduler](optimizer, warmup_end_steps=cfg.params.warmup_end_steps)
    #else:
    #    scheduler = None
    
    #if cfg.params.early_stopping_flag:
    #    early_stopper = am.EarlyStopping(patience=cfg.params.patience, verbose=True, delta=cfg.params.delta, path=os.path.join(log_model_path, "model.ckpt"))

    val_runner = Runner(val_loader, model, criterion,device)
    train_runner = Runner(train_loader, model, criterion, device,optimizer)

    if cfg.logger.wandb_flag is True: 
        tracker = WandBExperiment(cfg.paths.log_dir,cfg.params.project_name,cfg.models.model_name,cfg.params.optimizer,cfg.params.learning_rate,cfg.params.scheduler if cfg.params.scheduler is not None else "no")
    else:
        tracker = TensorboardExperiment(cfg.paths.log_dir,cfg.params.project_name,cfg.models.model_name,cfg.params.optimizer,cfg.params.learning_rate,cfg.params.scheduler if cfg.params.scheduler is not None else "no")
    
    for epoch in range(1, cfg.params.max_epochs+1):
        run_epoch(val_runner, train_runner,tracker, epoch)

        summary =",".join(
            [
                 f"[Epoch: {epoch}/{cfg.params.max_epochs}",
                f"  validation loss: {val_runner.avg_loss: 0.4f}",
                f"  Train loss: {train_runner.avg_loss: 0.4f}",
                f"  Validation Accuracy: {val_runner.avg_accuracy: 0.4f}",
                f"  Train Accuracy: {train_runner.avg_accuracy: 0.4f}",
            ]
        )
        print("\n" + summary + "\n")

        train_runner.reset()
        val_runner.reset()
        tracker.flush()


def main():
    pass

if __name__ == "__main__":
    main()
