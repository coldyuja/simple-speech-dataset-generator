import copy
from enum import Enum
import torch
from os import PathLike
from .vg_types import AudioSetting
from typing import Dict, Generic, NoReturn, Optional, TypedDict, Any
from .commons import AbstractPipelineElement
from .utils import get_torch_device
from torchaudio import sox_effects
from torch import Tensor
import torio
from torio.io import CodecConfig

# Cleaning Audio -> Noise Reduction
# https://github.com/timsainb/noisereduce    

class Result(TypedDict):
    reduce_noise: Any

class NoiseReductionModels(Enum):
    NOISEREDUCE = 'noisereduce'

class CleaningAudioSetting(TypedDict):
    reduce_noise_lib: NoiseReductionModels
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
            self.input = self.input.to(self.torch_device)
            self.ret['reduce_noise'] = tg(self.input)
        else:
            input = self.input.cpu().numpy()
            self.ret['reduce_noise'] = nr.reduce_noise(y=input, 
                                                       sr=self.settings['sr'],
                                                       stationary=self.opt_settings['stationary'],
                                                       )
        self.latest_task = 'reduce_noise'
        return
    
    def reduce_noise(self, lib: NoiseReductionModels | None = None) -> NoReturn:
        if lib is None:
            lib = self.opt_settings['reduce_noise_lib']
        match lib:
            case NoiseReductionModels.NOISEREDUCE:
                self._use_noisereduce()
            case _:
                raise ValueError(f'Unknown lib name: {lib}')
            
    
    def _execute(self):
        self.reduce_noise()
        if not torch.is_tensor(self.ret[self.latest_task]):
            self.ret[self.latest_task] = torch.tensor(self.ret[self.latest_task])
        return
    
    def get_result(self):
        return self.ret[self.latest_task]
        

class LoadAudioBackends(Enum):
    TORCH_SOX = 0
    LIBROSA = 1

class LoadAudioSettings(TypedDict):
    backend: LoadAudioBackends
    effects: Any
    channel_first: bool

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
        if isinstance(self.load_settings['effects'], list):
            e_channel_exist = False
            e_sr_exist = False
            for effect in self.load_settings['effects']:
                if len(effect) == 0:
                    continue
                if effect[0] == 'channels':
                    e_channel_exist = True
                if effect[0] == 'rate':
                    e_sr_exist = True

            if not e_channel_exist and self.settings['mono']:
                self.load_settings['effects'].append(['channels', '1'])
            if not e_sr_exist and self.settings['sr']:
                self.load_settings['effects'].append(['rate', str(self.settings['sr'])])

        wav, sr = sox_effects.apply_effects_file(self.input_path, self.load_settings['effects'], channels_first=self.load_settings['channel_first'])
        self.result = (wav, sr)
        return
    
    def _use_librosa(self):
        raise NotImplementedError()

class SaveAudioSettings(TypedDict):
    encoder: Optional[str] = None
    encoder_option: Optional[Dict[str, str]] = None
    encoder_sample_rate: Optional[int] = None
    encoder_num_channels: Optional[int] = None
    encoder_format: Optional[str] = None
    codec_config: Optional[CodecConfig] = None
    filter_desc: Optional[str] = None

class SaveAudioFiles(AbstractPipelineElement):
    def __init__(self, settings: AudioSetting):
        self.settings = settings
        self.enc_settings: SaveAudioSettings = settings['opt_settings']
    
    # list[audio: list[Tensor], out_path: list[str], new_settings: Optional[AudioSetting]] -> In
    def _process_input(self, input):

        if len(input) < 2:
            raise ValueError('Input must be given audio tensor and output path at least')
        elif len(input) >= 2:
            self.audio = input[0]
            self.out_path = input[1]
            if len(input) >= 3:
                self.settings = input[2]

        

    def _execute(self):
        for data, path in zip(self.audio, self.out_path):
            if self.settings.get('n_channels') is None:
                n_ch = 1
                if len(data.shape) == 2:
                    n_ch = data.shape[0]

                elif is_mono := self.settings.get('mono') is not None:
                    if not is_mono:
                        n_ch = 2

                self.settings['n_channels'] = n_ch

            encoder = torio.io.StreamingMediaEncoder(path)
            encoder.add_audio_stream(sample_rate=self.settings['sr'], num_channels=self.settings['n_channels'], **self.enc_settings)
            encoder.open()
            # torio.io.StreamingMediaEncoder.write_audio_chunk() Input shape: (S, C)
            if len(data.shape) == 1:
                data = data.unsqueeze(dim=-1)
            elif len(data.shape) == 2:
                data = data.transpose(1, 0)
            else:
                raise ValueError('Output Audio Tensor dimension size must be 1 or 2')
    
            for c_i in range(self.settings['n_channels']):
                encoder.write_audio_chunk(c_i, data[..., c_i].unsqueeze(dim=-1))
            encoder.flush()
            encoder.close()

    
    # Out -> None
    def get_result(self):
        return None
    






