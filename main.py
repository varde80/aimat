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
from hydra.utils import instantiate

#from omegaconf import DictConfig
import aimat
import aimat.models as models
from config import AIMatConfig
from aimat.logger import TensorboardExperiment
from aimat.logger import WandBExperiment
from aimat.core import Runner,run_epoch
from aimat.core import set_dataloader

cs = ConfigStore.instance()
cs.store(name="AIMat_config", node=AIMatConfig)

@hydra.main(config_path="conf", config_name="config2")
def main(cfg: AIMatConfig):
#def main(cfg):
#    print(cfg)    
    #check device mps or cuda or cpu 
    if "cuda" in cfg.device and torch.cuda. is_available():
        device = "cuda"
    elif "mps" in cfg.device and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}")
    
    #create dataloader
    train_loader, val_loader = set_dataloader(cfg)

    criterion = instantiate(cfg.criterion)
    model = instantiate(cfg.model).to(device)
    optimizer =instantiate(cfg.optimizer, params=model.parameters())
    #print(cfg.scheduler)
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer) if cfg.scheduler is not None else None    
        
    val_runner = Runner(val_loader, model, criterion,device)
    train_runner = Runner(train_loader, model, criterion, device,optimizer)

    if cfg.logger == 'wandb': 
        tracker = WandBExperiment(cfg)
    else:
        tracker = TensorboardExperiment(cfg)

    for epoch in range(1, cfg.max_epochs+1):
        run_epoch(val_runner, train_runner,tracker, epoch)

        summary =",".join(
            [
                 f"[Epoch: {epoch}/{cfg.max_epochs}",
                f"  validation loss: {val_runner.avg_loss: 0.4f}",
                f"  Train loss: {train_runner.avg_loss: 0.4f}",
                f"  Validation Accuracy: {val_runner.avg_accuracy: 0.4f}",
                f"  Train Accuracy: {train_runner.avg_accuracy: 0.4f}",
            ]
        )
        print("\n" + summary + "\n")

        ## save model
        if cfg.save_model and epoch % cfg.save_model_interval == 0:
            torch.save({
                        'epoch': epoch,
                        'model.state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_runner.avg_loss,
                        },os.path.join(cfg.paths.save_dir, f"model_{epoch}.pth")
                        )
        train_runner.reset()
        val_runner.reset()
        tracker.flush()
     
if __name__ == "__main__":
    main()
