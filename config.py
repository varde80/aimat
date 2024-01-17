from dataclasses import dataclass

@dataclass
class params:
    project_name: str
    device: str    
    max_epochs: int
    batch_size: int
    batch_size_test: int
    iscustom: bool

@dataclass
class paths:
    data_dir: str
    log_dir: str

@dataclass
class logger:
    wandb_flag: bool

@dataclass
class model:
    _target_: object


@dataclass
class dataset:
    dataset_name: str


@dataclass
class criterion:
    _target_: object

@dataclass
class optimizer:    
    _target_: object
    lr: float

@dataclass
class scheduler:
    _target_: object
    step_size: int
    gamma: float


@dataclass
class optuna:
    num_trials: int

@dataclass
class defaults:
    models: model
    dataset: dataset
    optimizer: optimizer
    scheduler: scheduler

@dataclass
class AIMatConfig:
    params: params    
    paths: paths    
    logger: logger
    defaults: defaults