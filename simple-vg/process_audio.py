import torch
from vg_types import PathLike, AudioSetting
from typing import Generic, NoReturn, TypedDict, Any
from commons import AbstractPipelineElement

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
            self.torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
        

    






