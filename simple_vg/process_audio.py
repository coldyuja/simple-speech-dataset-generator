import copy
from enum import Enum
import torch
from os import PathLike
from vg_types import AudioSetting
from typing import Generic, NoReturn, TypedDict, Any
from commons import AbstractPipelineElement
from utils import get_torch_device
from torchaudio import sox_effects
from torch import Tensor

# Cleaning Audio -> Noise Reduction
# https://github.com/timsainb/noisereduce    

class Result(TypedDict):
    reduce_noise: Any

class CleaningAudioSetting(TypedDict):
    reduce_noise_lib: str
    stationary: bool

class CleaningAudio(AbstractPipelineElement):
    def __init__(self, setting: AudioSetting):
        self.settings = setting  
        if self.settings['use_torch']:
            self.torch_device = get_torch_device()
        self.ret: Result = {}
        self.latest_task = None
        self.opt_settings: CleaningAudioSetting = setting['opt_settings']
        return
    
    def _process_input(self, input):
        self.input = input

    # https://github.com/timsainb/noisereduce
    # Features are not fully implemented currently.
    def _use_noisereduce(self) -> NoReturn:
        import noisereduce as nr
        from noisereduce.torchgate import TorchGate as TG
        if self.torch_device:
            tg = TG(sr=self.settings['sr'], 
                    nonstationary=not self.opt_settings['stationary']).to(self.torch_device)
            self.ret['reduce_noise'] = tg(self.input)
        else:
            self.ret['reduce_noise'] = nr.reduce_noise(y=self.input, 
                                                       sr=self.settings['sr'],
                                                       stationary=self.opt_settings['stationary'],
                                                       )
        self.latest_task = 'reduce_noise'
        return
    
    def reduce_noise(self, lib: str | None = None) -> NoReturn:
        if lib is None:
            lib = self.opt_settings['reduce_noise_lib']
        match lib:
            case 'noisereduce':
                self._use_noisereduce()
            case _:
                raise ValueError(f'Unknown lib name: {lib}')
            
    
    def _execute(self):
        self.reduce_noise()
        return
    
    def get_result(self):
        return self.ret[self.latest_task]
        

class LoadAudioBackends(Enum):
    TORCH_SOX = 0
    LIBROSA = 1

class LoadAudioSettings(TypedDict):
    backend: LoadAudioBackends
    effects: Any

class LoadAudioFile(AbstractPipelineElement):
    def __init__(self, settings: AudioSetting):
        settings = copy.deepcopy(settings)
        self.settings = settings
        self.load_settings: LoadAudioSettings = settings['opt_settings']
        return
    
    def _process_input(self, input):
        if not isinstance(input, (str, PathLike)):
            raise ValueError('input must be str or PathLike containing file path')
        
        self.input_path = input
    
    def _execute(self):
        backend = self.load_settings['backend']
        match backend:
            case LoadAudioBackends.TORCH_SOX:
                self._use_torch_sox()
            case LoadAudioBackends.LIBROSA:
                self._use_librosa()
            case _:
                raise KeyError(f'Unknown backend name: {backend}')
                
    
    def get_result(self) -> tuple[Tensor, int]:
        return self.result
    
    def _use_torch_sox(self):
        wav, sr = sox_effects.apply_effects_file(self.input_path, self.load_settings['effects'])
        self.result = (wav, sr)
        return
    
    def _use_librosa(self):
        raise NotImplementedError()







