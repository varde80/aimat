import torch
import torchvision
import torchvision.transforms as transforms
from hydra.utils import instantiate

def set_dataloader(cfg):
    
    iscustom:bool=False if cfg.iscustom is None else cfg.iscustom

    if iscustom is False:
        train_data = torchvision.datasets.__dict__[cfg.dataset.dataset](root=cfg.paths.data_dir, train=True, download=True, transform=transforms.ToTensor())
        test_data = torchvision.datasets.__dict__[cfg.dataset.dataset](root=cfg.paths.data_dir, train=False, download=True, transform=transforms.ToTensor())
        return torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True,),\
               torch.utils.data.DataLoader(test_data, batch_size=cfg.batch_size_test,shuffle=False,)
