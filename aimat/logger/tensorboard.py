import os
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from aimat.core import Stage

class TensorboardExperiment:
    def __init__(
            self,
            cfg,
            #root_dir: str = cfg.paths.log_dir,
            #project_name: str,
            #model_name: str,            
            #optimizer_name: str, 
            #learning_rate: float,
            #scheduler_name: str,
            create: bool = True
            ):
        
        root_dir = cfg.paths.log_dir
        project_name = cfg.project_name
        
        model_name = (cfg.model._target_).split(".")[-1]
        optimizer_name = (cfg.optimizer._target_).split(".")[-1]
        learning_rate = cfg.optimizer.lr
        scheduler_name = (cfg.scheduler._target_).split(".")[-1] if cfg.scheduler is not None else "no"
    

        run_name = f"{datetime.now().isoformat(timespec='seconds')}-{model_name}-{optimizer_name}_optim_{learning_rate}_lr_with_{scheduler_name}_scheduler"
        log_dir = f"{root_dir}/{run_name}"
        os.makedirs(log_dir, exist_ok=create)

        self.stage= Stage.TRAIN
        self._validate_log_dir(log_dir,create=create)
        self.writer = SummaryWriter(log_dir=log_dir)
        
    def set_stage(self, stage: Stage):
        self.stage = stage

    def flush(self):
        self.writer.flush()
    
    @staticmethod
    def _validate_log_dir(log_dir: str,create: bool = True):
        log_path = Path(log_dir).resolve()
        if log_path.exists():
            return
        elif not log_path.exist() and create:
            log_path.mkdir(parents=True)
        else:
            raise ValueError(f"log_dir {log_dir} does not exist.")
    
    def add_batch_metric(self, name: str, value: float, step: int, commit:bool):
        tag = f"{self.stage.name}/batch/{name}"
        self.writer.add_scalar(tag, value, step)        

    def add_epoch_metric(self, name: str, value: float, step: int,commit:bool):
        tag = f"{self.stage.name}/epoch/{name}"
        self.writer.add_scalar(tag, value, step)
 