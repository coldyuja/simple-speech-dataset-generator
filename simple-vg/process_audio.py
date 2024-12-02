import torch
from vg_types import PathLike, AudioSetting
from typing import NoReturn, TypedDict, Any


# Detect Overlapped Speech


# Cleaning Audio -> Noise Reduction
# https://github.com/timsainb/noisereduce    

class Result(TypedDict):
    reduce_noise: Any

class CleaningAudio:
    def __init__(self, input, setting: AudioSetting):
        self.setting = setting
        self.input = input
        if self.setting['use_torch']:
            self.torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.ret: Result = {}
        self.latest_task = None
        return
    

    # https://github.com/timsainb/noisereduce    
    def _use_noisereduce(self) -> NoReturn:
        import noisereduce as nr
        from noisereduce.torchgate import TorchGate as TG
        if self.torch_device:
            tg = TG(sr=self.setting['sr'], 
                    nonstationary=not self.setting['opt_settings']['stationary']).to(self.torch_device)
            self.ret['reduce_noise'] = tg(self.input)
        else:
            self.ret['reduce_noise'] = nr.reduce_noise(y=self.input, 
                                                       sr=self.setting['sr'],
                                                       stationary=self.setting['opt_settings']['stationary'],
                                                       )
        self.latest_task = 'reduce_noise'
        return
    
    def reduce_noise(self, lib: str = 'reducenoise'):
        match lib:
            case 'noisereduce':
                self._use_noisereduce()
            case _:
                raise ValueError(f'Unknown lib name: {lib}')
            
    def get_final_result(self):
        return self.ret[self.latest_task]

    






