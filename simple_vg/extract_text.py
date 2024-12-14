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
from .stt.whisper_wrapper import WhisperSettings, WhisperWrapper, WhisperTasks, DecodingOptions
from .utils import fill_dict

class ExtractTextModels(Enum):
    WHISPER_TINY = 'tiny'
    WHISPER_TINY_EN = 'tiny.en'
    WHISPER_BASE = 'base'
    WHISPER_BASE_EN = 'base.en'
    WHISPER_SMALL = 'small'
    WHISPER_SMALL_EN = 'small.en'
    WHISPER_MEDIUM = 'medium'
    WHISPER_MEDIUM_EN = 'medium.en'
    WHISPER_LARGE = 'large'
    WHISPER_LARGE_V1 = 'large-v1'
    WHISPER_LARGE_V2 = 'large-v2'
    WHISPER_LARGE_V3 = 'large-v3'
    WHISPER_TURBO_V3 = 'large-v3-turbo'
    WHISPER_TURBO = 'turbo'

class ExtractTextSettings(TypedDict):
    model: ExtractTextModels
    verbose: bool
    batch_size: int
    model_settings: Any
    max_chunk_len: int | None


class ExtractText(AbstractPipelineElement):
    def __init__(self, settings: AudioSetting):
        self.settings = copy.deepcopy(settings)
        self.model_settings: ExtractTextSettings = self.settings['opt_settings']
        fill_dict(self.model_settings, self.model_settings['model_settings'], set(['batch_size', 'model']))

        if 'WHISPER' in self.model_settings['model'].name:
            self.model = WhisperWrapper(self.model_settings['model_settings'])
        return
    
    def _process_input(self, input):
        # Input Tensor shape: (B, C, S) or list[(C, S)] 
        self.input = input

    def _execute(self):
        self.model.process_input(self.input)
        self.model.inference()
        self.result = self.model.get_result()
        return
    
    def get_result(self):
        return self.result
    

def default_settings() -> ExtractTextSettings:
    w_settings: WhisperSettings = {
        'task': WhisperTasks.TRANSCRIBE,
        'decode_options': DecodingOptions()
    }
    settings: ExtractTextSettings = {
        'batch_size': 1,
        'max_chunk_len': None,
        'model': ExtractTextModels.WHISPER_TURBO,
        'model_settings': w_settings,
    }

    return settings

    