from torch.utils.data import DataLoader
from typing import Any, Optional
import torch
from aimat.core import Metric, Tracker, Stage
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

class Runner:
    def __init__(
        self,
        loader: DataLoader[Any],
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        device: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        ) -> None:
        self.metric_loss = Metric()
        self.metric_accuracy = Metric()
        self.train_step = 0
        self.loader = loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        self.stage = Stage.VAL if optimizer is None else Stage.TRAIN 

    @property
    def avg_accuracy(self):
        return self.metric_accuracy.average
    
    @property
    def avg_loss(self):
        return self.metric_loss.average
    
    def run(self, str, experiment:Tracker):
        self.model.train(self.stage is Stage.TRAIN)

        for data, label in tqdm(self.loader, position=0, leave=True, desc=f"{self.stage.name}"):
            data, label = data.to(self.device), label.to(self.device)
            loss, batch_accuracy = self._run_single_batch(data,label)

            #experiment.add_batch_metric("accuracy", batch_accuracy, self.train_step,commit=False)
            #experiment.add_batch_metric("loss", loss, self.train_step,commit=True)
            
            if self.optimizer is not None:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
    def _run_single_batch(self, data:Any, label:Any):
        self.train_step += 1
        batch_size: int = data.shape[0]
        pred = self.model(data)
        loss = self.criterion(pred,label)
        
        label_np = label.detach().cpu().numpy()
        label_pred_np = np.argmax(pred.detach().cpu().numpy(), axis=1)
        batch_accuracy:float = accuracy_score(label_np,label_pred_np)

        self.metric_accuracy.update(batch_accuracy, batch_size)
    
        self.metric_loss.update(loss.item(), batch_size)
        return loss, batch_accuracy
    
    def reset(self):
        self.metric_loss = Metric()
        self.metric_accuracy = Metric()

def run_epoch(
    val_runner: Runner,
    train_runner: Runner,
    experiment: Tracker,
    epoch_id: int,
):
    experiment.set_stage(Stage.TRAIN)
    train_runner.run("train", experiment)

    experiment.add_epoch_metric("loss", train_runner.avg_loss, epoch_id,commit=False)
    experiment.add_epoch_metric("accuracy", train_runner.avg_accuracy, epoch_id,commit=False)

    experiment.set_stage(Stage.VAL)
    val_runner.run("validation", experiment)

    experiment.add_epoch_metric("loss", val_runner.avg_loss, epoch_id,commit=False)
    experiment.add_epoch_metric("accuracy", val_runner.avg_accuracy, epoch_id,commit=True)

