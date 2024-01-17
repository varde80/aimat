from enum import Enum, auto
from typing import Protocol

class Stage(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()

class Tracker(Protocol):
    def set_stage(self, stage: Stage):
        """Sets the current stage of the experiment."""

    def add_batch_metric(self, name: str, value: float, step: int):
        """Implements logging a batch-level metric."""

    def add_epoch_metric(self, name: str, value: float, step: int):
        """Implements logging a epoch-level metric."""
