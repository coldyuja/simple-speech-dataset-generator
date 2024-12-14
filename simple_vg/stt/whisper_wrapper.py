import copy
from enum import Enum
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, TypedDict, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
import whisper
import torch.nn.functional as F
import math

from simple_vg.utils import AttributeDummyClass, get_torch_device
from simple_vg.commons import ModelWrapper

class WhisperModels(Enum):
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

class WhisperTasks(Enum):
    TRANSCRIBE = 'transcribe' # X -> X
    TRANSLATE = 'translate' # X -> EN

# From whisper/whisper/decoding.py
class DecodingOptions(TypedDict):
    task: str
    language: Optional[str] = None
    temperature: float = 0.0
    sample_len: Optional[int] = None  # maximum number of tokens to sample
    best_of: Optional[int] = None  # number of independent sample trajectories, if t > 0
    beam_size: Optional[int] = None  # number of beams in beam search, if t == 0
    patience: Optional[float] = None  # patience in beam search (arxiv:2204.05424)
    length_penalty: Optional[float] = None
    prompt: Optional[Union[str, List[int]]] = None  # for the previous context
    prefix: Optional[Union[str, List[int]]] = None  # to prefix the current context
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    suppress_blank: bool = True  # this will suppress blank outputs
    without_timestamps: bool = False  # use <|notimestamps|> to sample text tokens only
    max_initial_timestamp: Optional[float] = 1.0
    fp16: bool = True  # use fp16 for most of the calculation

class WhisperSettings(TypedDict):
    decode_options: DecodingOptions
    model: WhisperModels
    task: WhisperTasks
    n_mels: int = None
    batch_size: int

#From whisper/whisper/decoding.py
class DecodingResult(TypedDict):
    audio_features: Tensor
    language: str
    language_probs: Optional[Dict[str, float]] 
    tokens: List[int]
    text: str
    avg_logprob: float
    no_speech_prob: float 
    temperature: float
    compression_ratio: float

class WhisperDataset(Dataset):
    def __init__(self, input: Tensor | list[Tensor], settings: WhisperSettings):
        self.raw_input = input
        self.settings = settings
        self.max_len = 3000
        self.indice = []

        if torch.is_tensor(self.raw_input):    
            if len(self.raw_input.shape) > 2:
                s = self.raw_input.shape
                self.raw_input = self.raw_input.view(math.prod(s[:-1]), s[-1])
            self.input = whisper.log_mel_spectrogram(self.raw_input, self.settings['n_mels'])
        elif isinstance(self.raw_input, (list)):
            if len(self.raw_input) == 0:
                raise ValueError('Input cannot empty!')

            if len(self.raw_input[0].shape) > 2:
                raise ValueError(f'Tensor elements in list must have 1-dimension. Current shape: {self.raw_input[0].shape}')
            self.input = [whisper.log_mel_spectrogram(inp, self.settings['n_mels']) for inp in self.raw_input]

        self._calc_indice_data()


    def _calc_indice_data(self):
        data_len = len(self.input)
        for i in range(data_len):
            data_indice = self._calc_indice_single_tensor(self.input[i], i)
            self.indice += data_indice
        return
    
    def _calc_indice_single_tensor(self, input, data_idx) -> list[list[int]]:
        max_len = self.max_len
        sample_len = input.shape[-1]
        ret = [[data_idx, max_len * i, max_len * i+1] for i in range(int(sample_len/max_len))]
        if len(ret) and ret[-1][2] < sample_len:
            ret.append([data_idx, ret[-1][1], sample_len])
        elif len(ret) == 0:
            ret.append([data_idx, 0, sample_len])
        return ret

    def __len__(self):
        return len(self.indice)
    
    def __getitem__(self, index):
        data_i, sample_st, sample_ed = self.indice[index]
        return (data_i, self.input[data_i][..., sample_st:sample_ed])
    

class WhisperWrapper(ModelWrapper):
    def __init__(self, settings: WhisperSettings):
        self.settings = copy.deepcopy(settings)
        self.model = whisper.load_model(self.settings['model'].value)
        self.settings['n_mels'] = self.model.dims.n_mels
        self.settings['decode_options']['task'] = self.settings['task'].value
        self.settings['decode_options'].setdefault('temperature', 0.0)
        self.settings['decode_options'].setdefault('suppress_tokens', '-1')
        self.settings['decode_options'].setdefault('suppress_blank', True)
        self.settings['decode_options'].setdefault('without_timestamps', False)
        self.settings['decode_options'].setdefault('max_initial_timestamp', 1.0)
        self.settings['decode_options'].setdefault('fp16', True)

        self.device = get_torch_device()

        
    def process_input(self, input: Tensor | list[Tensor]):
        self.dataset = WhisperDataset(input, self.settings)
        self.dataloader = DataLoader(self.dataset, self.settings['batch_size'], collate_fn=_collate)


    def inference(self):
        ret_list = [[] for _ in range(len(self.dataset))]
        decode_options = AttributeDummyClass(self.settings['decode_options'])
        for data_indice, mels in self.dataloader:
            mels = mels.to(self.device)
            ret = whisper.decode(self.model, mels, decode_options)
            for data_i, single_data in zip(data_indice, ret):
                if isinstance(single_data, (list)):
                    ret_list[data_i].append(*single_data)
                else:
                    ret_list[data_i].append(single_data)
        self.result = ret_list
        return
    
    def get_result(self) -> list[list[DecodingResult]]:

        return self.result
        
def _collate(inp):
    data_i = [single_sample[0] for single_sample in inp]
    data = [single_sample[1] for single_sample in inp]
    data = pad_sequence(data, batch_first=True)
    data_len = data.shape[-1]
    if data_len < 3000:
        data = F.pad(data, (0,3000-data_len))
        
    return data_i, data
