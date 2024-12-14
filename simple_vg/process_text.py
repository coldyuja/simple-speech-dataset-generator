import copy
from enum import Enum
import sys
from typing import Any, TypedDict
from torch import Tensor
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from .commons import AbstractPipelineElement
from .vg_types import AudioSetting
from .stt.whisper_wrapper import WhisperSettings, WhisperWrapper, WhisperTasks
from .utils import fill_dict


class SaveAsDataset(AbstractPipelineElement):
    def __init__(self):
        return
    
    def _process_input(self, ):
        return super()._process_input(input)
    
    def _execute(self):
        return super()._execute()
    
    def get_result(self):
        return None