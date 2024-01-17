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

import wandb
import optuna
from optuna.trial import TrialState
from optuna.integration.wandb import WeightsAndBiasesCallback

from aimat.core.tracker import Tracker,Stage
#import wandb

import hydra
from hydra.core.config_store import ConfigStore 
from hydra.utils import instantiate

#from omegaconf import DictConfig

import aimat.core as am
import aimat.models as models
from config import AIMatConfig
from aimat.logger.tensorboard import TensorboardExperiment, WandBExperiment
from aimat.core.runner import Runner,run_epoch

cs = ConfigStore.instance()
cs.store(name="AIMat_config", node=AIMatConfig)

'''
wandb_kwargs = {
    "project": "optuna-v3-wandb-pytorch",
    "entity": "kims-ai",
    "reinit": True,
}

wandbc = WeightsAndBiasesCallback(
    metric_name="final validation accuracy", wandb_kwargs=wandb_kwargs, as_multirun=True
)


@wandbc.track_in_wandb()
'''
def objective(
        trial,
        train_runner: Runner,
        val_runner: Runner,
        tracker: Tracker,
        max_epochs: int,
        ):

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)                                 # Learning rates

    for epoch in range(1, max_epochs+1):
        tracker.set_stage(Stage.TRAIN)
        train_runner.run("train", tracker)

        tracker.add_epoch_metric("loss", train_runner.avg_loss, epoch,commit=False)
        tracker.add_epoch_metric("accuracy", train_runner.avg_accuracy, epoch,commit=False)

        tracker.set_stage(Stage.VAL)
        val_runner.run("validation", tracker)

        tracker.add_epoch_metric("loss", val_runner.avg_loss, epoch,commit=False)
        tracker.add_epoch_metric("accuracy", val_runner.avg_accuracy, epoch,commit=True)

        accuracy = val_runner.avg_accuracy

        summary =",".join(
            [
                 f"[Epoch: {epoch}/{max_epochs}",
                f"  validation loss: {val_runner.avg_loss: 0.4f}",
                f"  Train loss: {train_runner.avg_loss: 0.4f}",
                f"  Validation Accuracy: {val_runner.avg_accuracy: 0.4f}",
                f"  Train Accuracy: {train_runner.avg_accuracy: 0.4f}",
            ]
        )
        print("\n" + summary + "\n")

        trial.report(accuracy,epoch)
        if trial.should_prune():
            #wandb.run.summary["state"] = "pruned"
            raise optuna.exceptions.TrialPruned()

        train_runner.reset()
        val_runner.reset()
        tracker.flush()

    #wandb.run.summary["final accuracy"] = accuracy
    #wandb.run.summary["state"] = "completed"
    return accuracy     

@hydra.main(config_path="conf", config_name="config")
def main(cfg: AIMatConfig):
    
    #check device mps or cuda or cpu 
    if "cuda" in cfg.params.device and torch.cuda. is_available():
        device = "cuda"
    elif "mps" in cfg.params.device and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"device: {device}")
    
    #create dataloader
    train_loader, val_loader = am.set_dataloader(cfg)

    criterion = instantiate(cfg.criterion)
    model = instantiate(cfg.model).to(device)
    optimizer =instantiate(cfg.optimizer, params=model.parameters())

    #if cfg.params.scheduler is not None: 
    #    scheduler = am.__dict__[cfg.params.scheduler](optimizer, warmup_end_steps=cfg.params.warmup_end_steps)
    #else:
    #    scheduler = None
    
    #if cfg.params.early_stopping_flag:
    #    early_stopper = am.EarlyStopping(patience=cfg.params.patience, verbose=True, delta=cfg.params.delta, path=os.path.join(log_model_path, "model.ckpt"))

    val_runner = Runner(val_loader, model, criterion,device)
    train_runner = Runner(train_loader, model, criterion, device,optimizer)

    if cfg.logger.wandb_flag is True: 
        tracker = WandBExperiment(cfg)
    else:
        tracker = TensorboardExperiment(cfg)

    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_runner=train_runner, val_runner=val_runner, tracker=tracker,max_epochs=cfg.params.max_epochs ), n_trials=cfg.optuna.num_trials)#, callbacks=[wandbc])

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Display the study statistics
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save results to csv file
    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
    df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
    df = df.drop('state', axis=1)                 # Exclude state column
    df = df.sort_values('value')                  # Sort based on accuracy
    df.to_csv('optuna_results.csv', index=False)  # Save to csv file

    # Display results in a dataframe
    print("\nOverall Results (ordered by accuracy):\n {}".format(df))


if __name__ == "__main__":
    main()
