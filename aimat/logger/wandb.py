import os
from datetime import datetime
from pathlib import Path
from aimat.core import Stage
import wandb

class WandBExperiment:
    def __init__(
            self,
            cfg,
            create: bool = True
            ):
        
        root_dir = cfg.paths.log_dir
        project_name = cfg.project_name
        model_name = (cfg.model._target_).split(".")[-1]
        optimizer_name = (cfg.optimizer._target_).split(".")[-1]
        learning_rate = cfg.optimizer.lr
        scheduler_name = (cfg.scheduler._target_).split(".")[-1] if cfg.scheduler is not None else "no"
    
        run_name = f"{datetime.now().isoformat(timespec='seconds')}-{model_name}-{optimizer_name}_optim_{learning_rate}_lr_with_{scheduler_name}_scheduler"
        run_tags = [project_name]
        self.stage= Stage.TRAIN
        wandb.init(
            project = project_name,
            name=run_name,
            tags=run_tags,
            config={"lr": learning_rate, "model_name": model_name, "optimizer_name": optimizer_name, "scheduler_name": scheduler_name},
            reinit=True,
        )       
        
    def set_stage(self, stage: Stage):
        self.stage = stage

    def flush(self):
        pass
        
    def add_batch_metric(self, name: str, value: float, step: int,commit:bool):
        tag = f"{self.stage.name}/batch/{name}"
        wandb.log({tag: value},step = step,commit=commit)

    def add_epoch_metric(self, name: str, value: float, step: int,commit:bool):
        tag = f"{self.stage.name}/epoch/{name}"
        wandb.log({tag: value},step = step, commit=commit)
        