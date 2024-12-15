# 해당 파일에서 사용할 모듈들을 Import함.
import copy
from enum import Enum
import torch
from os import PathLike
from .vg_types import AudioSetting
from typing import Dict, NoReturn, Optional, TypedDict, Any
from .commons import AbstractPipelineElement
from .utils import get_torch_device
from torchaudio import sox_effects
from torch import Tensor
import torio
from torio.io import CodecConfig
import noisereduce as nr
from noisereduce.torchgate import TorchGate as TG

# 여기서 사용하는 noisereduce 모듈의 github 주소이다.
# https://github.com/timsainb/noisereduce    

# 해당 Dict가 사용되는 Class가 CleaningAudio인데 
# 추후 Noise Reduction 이외 다른 방식도 추가 예정이기 때문에 결과를 위한 Dict를 정의했음.
# vg_types.py에서 설명했지만, TypedDict을 사용하면 Key나 Value의 Type을 코드 작성시 특정 개발 도구에서 볼수 있기 때문에
# 오타율이 적고 빠르게 코드 작성이 가능해짐.
class Result(TypedDict):
    reduce_noise: Any

# Noise Reduction에 사용하는 모듈이나 모델을 명시해놓은 Enum임.
# Enum은 TypedDict과 같이 코드 작성 시 어떤 내용이 들어갈 수 있는지 확인도 가능하고, 자동완성으로 단어를 전부 작성할 필요가 없어 오타율이 감소함.
class NoiseReductionModels(Enum):
    NOISEREDUCE = 'noisereduce'

# CleaningAudio class에서 사용하는 Setting class임.
class CleaningAudioSetting(TypedDict):
    reduce_noise_lib: NoiseReductionModels
    # Noise가 Stationary한지 여부를 명시할 수 있음.
    stationary: bool

# Pipeline에서 사용할 Element들은 AbstractPipelineElement를 부모 클래스로 가져야 한다.
# AbstractPipelineElement에 있는 method들을 통해 실행되기 때문에 해당 Element들은 그 method들을
# 각 Class에서 구현하여 Override해야 한다. 그렇지 않으면 NotImplementError가 발생한다.
class CleaningAudio(AbstractPipelineElement):

    # 해당 class를 초기화 한다.
    # 주어진 setting으로 나중에 실행될 설정들을 class 내에 저장한다.
    def __init__(self, setting: AudioSetting):
        self.settings = setting  
        if self.settings['use_torch']:
            self.torch_device = get_torch_device()
        self.ret: Result = {}
        self.latest_task = None
        self.opt_settings: CleaningAudioSetting = setting['opt_settings']
        return
    
    # 그대로 input을 저장함.
    # 입력: Tensor
    def _process_input(self, input):
        self.input: Tensor = input

    # https://github.com/timsainb/noisereduce
    # 여기서 noisereduce가 사용됨.
    def _use_noisereduce(self) -> NoReturn:
        # torch 사용시 torch를 사용하도록 한다. 또한 CleaningAudioSetting에 있던 설정들을 가져와 적용한다.
        if self.torch_device:
            tg = TG(sr=self.settings['sr'], 
                    nonstationary=not self.opt_settings['stationary']).to(self.torch_device)
            self.input = self.input.to(self.torch_device)
            self.ret['reduce_noise'] = tg(self.input)
        # nr.redice_noise가 ndarray를 받기 때문에 아니라면 Input Tensor을 numpy의 ndarray로 변경한다.
        else:
            input = self.input.cpu().numpy()
            self.ret['reduce_noise'] = nr.reduce_noise(y=input, 
                                                       sr=self.settings['sr'],
                                                       stationary=self.opt_settings['stationary'],
                                                       )
        # 여러 Task를 해당 Class에서 할 수 있도록 구현할 예정이기 때문에
        # 마지막 task가 무엇인지 따로 저장한다. 마지막 출력을 리턴하는 get_result()에서 사용된다.
        self.latest_task = 'reduce_noise'
        return
    
    # NoiseReductionModels에 따라 각 코드로 분기된다.
    def reduce_noise(self, lib: NoiseReductionModels | None = None) -> NoReturn:
        if lib is None:
            lib = self.opt_settings['reduce_noise_lib']
        match lib:
            case NoiseReductionModels.NOISEREDUCE:
                self._use_noisereduce()
            case _:
                raise ValueError(f'Unknown lib name: {lib}')
            
    # Pipeline에서 사용할 method임. 해당 method를 호출하여 실행한다.
    def _execute(self):
        self.reduce_noise()
        if not torch.is_tensor(self.ret[self.latest_task]):
            self.ret[self.latest_task] = torch.tensor(self.ret[self.latest_task])
        return
    
    # Pipeline에서 사용할 method임. 최종 결과를 리턴함. 
    def get_result(self):
        return self.ret[self.latest_task]
        

# Audio를 불러올때 사용할 백엔드 선택을 위한 Enum
# Dict, Enum에 관한 설명은 위에서 하였음.
class LoadAudioBackends(Enum):
    TORCH_SOX = 0

# LoadAudioFile에서 사용할 Setting Dict임
class LoadAudioSettings(TypedDict):
    backend: LoadAudioBackends
    effects: list[list[str]]
    channel_first: bool

# 주어진 Path에 있는 AudioFile을 Tensor로 변환하고 effect를 적용한다.
# effect 적용과 Tensor로의 변환은 torchaudio.sox_effets를 사용한다.  
class LoadAudioFile(AbstractPipelineElement):
    def __init__(self, settings: AudioSetting):
        # 기본적으로 AudioSetting에 opt_settings에 각 Class 별 Setting을 넣어오기 때문에
        # AudioSetting 자체는 돌려쓰게 될수 있다. 따라서 추후 변경될 수 있기 때문에 deepcopy 수행한다.
        settings = copy.deepcopy(settings)
        self.settings = settings
        self.load_settings: LoadAudioSettings = settings['opt_settings']
        return
    
    # 입력: str | PathLike
    def _process_input(self, input):
        if not isinstance(input, (str, PathLike)):
            raise ValueError('input must be str or PathLike containing file path')
        
        self.input_path = input
    
    # LoadAudioBackends에 맞추어 분기하여 작업 수행한다.
    def _execute(self):
        backend = self.load_settings['backend']
        match backend:
            case LoadAudioBackends.TORCH_SOX:
                self._use_torch_sox()
            case _:
                raise KeyError(f'Unknown backend name: {backend}')
            
    # C=Channel, S=Sample 
    # 출력: [Tensor(C, S), SamplingRate]
    def get_result(self) -> tuple[Tensor, int]:
        return self.result
    
    def _use_torch_sox(self):
        if isinstance(self.load_settings['effects'], list):
            e_channel_exist = False
            e_sr_exist = False

            # 이미 channels 설정과 rate설정이 존재하는지 체크함.
            for effect in self.load_settings['effects']:
                if len(effect) == 0:
                    continue
                if effect[0] == 'channels':
                    e_channel_exist = True
                if effect[0] == 'rate':
                    e_sr_exist = True

            # 없는 설정은 여기서 따로 적용함.
            if not e_channel_exist and self.settings['mono']:
                self.load_settings['effects'].append(['channels', '1'])
            if not e_channel_exist and self.settings['n_channels']:
                self.load_settings['effects'].append(['channels', str(self.settings['n_channels'])])
            if not e_sr_exist and self.settings['sr']:
                self.load_settings['effects'].append(['rate', str(self.settings['sr'])])

        # 여기서 최종적으로 Effect를 적용하고 Tensor와 sr를 받음.
        wav, sr = sox_effects.apply_effects_file(self.input_path, self.load_settings['effects'], channels_first=self.load_settings['channel_first'])
        self.result = (wav, sr)
        return
    
# torio.io.StreamingMediaEncoder.add_audio_stream 파라미터 그대로 가져왔음.
class SaveAudioSettings(TypedDict):
    encoder: Optional[str] = None
    encoder_option: Optional[Dict[str, str]] = None
    encoder_sample_rate: Optional[int] = None
    encoder_num_channels: Optional[int] = None
    encoder_format: Optional[str] = None
    codec_config: Optional[CodecConfig] = None
    filter_desc: Optional[str] = None

# Tensor로 된 Audio를 파일로 저장함.
class SaveAudioFiles(AbstractPipelineElement):
    def __init__(self, settings: AudioSetting):
        self.settings = settings
        self.enc_settings: SaveAudioSettings = settings['opt_settings']
    
    # 입력: list[audio: list[Tensor((C, S) or (S,))], out_path: list[str], new_settings: Optional[AudioSetting]]
    def _process_input(self, input):
        # audio와, out_path는 필수적으로 제공되야 하므로 input의 길이로 이를 확인함.
        if len(input) < 2:
            raise ValueError('Input must be given audio tensor and output path at least')
        elif len(input) >= 2:
            self.audio = input[0]
            self.out_path = input[1]

            # new_settings가 있다면 이를 적용.
            if len(input) >= 3:
                self.settings = input[2]

        

    def _execute(self):
        for data, path in zip(self.audio, self.out_path):
            # 각 데이터마다 Channel 수가 다를수도 있으므로 이를 찾기 위해 아래의 동작을 수행함.
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
            # Input Tensor shape에 맞추어 Tensor(S) 인 경우 (S, 1)로 만들어줌. 
            if len(data.shape) == 1:
                data = data.unsqueeze(dim=-1)

            # 해당 함수의 Tensor Input shape가 (S, C)이기 때문에 transpose함.
            elif len(data.shape) == 2:
                data = data.transpose(1, 0)
            else:
                raise ValueError('Output Audio Tensor dimension size must be 1 or 2')
            
            # 오디오 채널이 있는 만큼 각 Tensor을 각 Channel에 write함.
            for c_i in range(self.settings['n_channels']):
                encoder.write_audio_chunk(c_i, data[..., c_i].unsqueeze(dim=-1))
            encoder.flush()
            encoder.close()

    
    # Out -> None
    # 리턴값이 따로 없는 Element임.
    def get_result(self):
        return None
    






