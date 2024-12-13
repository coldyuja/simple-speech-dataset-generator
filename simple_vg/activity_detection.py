import copy
import torch
from torch import Tensor
from torchaudio import sox_effects
from .vg_types import AudioSetting
from typing import Generic, NewType, NoReturn, Any, TypedDict, AnyStr, Union
from .commons import AbstractPipelineElement
from enum import Enum

#Split by Voice Activity Detection

class VoiceActivityDetectionTasks(Enum):
    ACTIVITY_DETECTION = 1
    SPLIT_BY_ACTIVITY = 2

class VADModels(Enum):
    Silero_VAD = 0

class VADPipelineSetting(TypedDict):
    vad_model: VADModels

class VoiceActivityDetectionSetting(TypedDict):
    tasks: list[VoiceActivityDetectionTasks]
    pipeline_settings: VADPipelineSetting | None

class Timestamp(TypedDict):
    start: int | float
    end: int | float

class TimestampUnit(Enum):
    SAMPLE = 1
    SEC = 2
    MSEC = 3


class VoiceActivityDetection(AbstractPipelineElement):
    def __init__(self, settings: AudioSetting):
        self.settings = copy.deepcopy(settings)
        self.opt_settings: VoiceActivityDetectionSetting = self.settings['opt_settings']

    
    def _process_input(self, input):
        # input_tensor shape: [channels, samples]
        if isinstance(input, (torch.Tensor)):
            self.input_tensors: list[Tensor] = [input]
        elif isinstance(input, (list[Tensor])):
            self.input_tensors = input
        else:
            raise ValueError('Input type must be Tensor or list[Tensor]')
        
    #silero-vad (MIT License)
    def _use_silero_vad(self) -> NoReturn:
        # silero_vad use window_size=512 for 16KHz sr, 256 for 8KHz sr
        self.sr = self.settings['sr']
        if self.sr:
            if self.sr % 16000 == 0:
                self.window_size = 512
            elif self.sr == 8000:
                self.window_size = 256
            else:
                raise ValueError('Not Supported Sampling Rate. Please make it 8000 or 16000*n')
        else:
            self.window_size = 512
            self.sr = 16000

        # get_speech_timestamps needs to take input which is shaped as 1D Tensor
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        get_speech_timestamps, _, _, _, _ = utils

        self.timestamps: list[list[Timestamp]] = []
        self.timestamp_units: list[TimestampUnit] = []

        for data in self.input_tensors:
            wav = data.squeeze(0)
            self.timestamps.append(get_speech_timestamps(wav, model))
            self.timestamp_units.append(TimestampUnit.SAMPLE)

        return
    
    def detect(self, model_name: VADModels = VADModels.Silero_VAD) -> int:
        match model_name:
            case VADModels.Silero_VAD:
                self._use_silero_vad()
            case _:
                raise ValueError(f'Unknown model_name: {model_name}')

        self.latest_ret = self.timestamps
        return len(self.timestamps)

    # if you want to override timestamps, put new timestamps into below param.
    def split_audio(self, timestamps_list: list[list[Timestamp]] = None, timestamp_units: list[TimestampUnit]=None) -> NoReturn:
        if timestamps_list is None:
            timestamps_list = self.timestamps
        if timestamp_units is None:
            timestamp_units = self.timestamp_units

        self.splitted_audio = []

        for i in range(len(timestamps_list)):
            curr_splitted_audio =  []
            for ts in timestamps_list[i]:
                st, end = ts['start'], ts['end']

                # location of sample at t-sec = t * sr
                # Sampling Rate(sr) => the number of samples in 1 duration sec
                if timestamp_units[i] == TimestampUnit.SEC:
                    st = st * self.sr
                    end = end * self.sr
                elif timestamp_units[i] == TimestampUnit.MSEC:
                    st = st * 1000 * self.sr
                    end = end * 1000 * self.sr

                curr_splitted_audio.append(self.input_tensors[i][..., st:end+1])
            self.splitted_audio.append(curr_splitted_audio)

        self.latest_ret = self.splitted_audio             
        return

    def _execute(self):
        for task in self.opt_settings['tasks']:
            match task:
                case VoiceActivityDetectionTasks.ACTIVITY_DETECTION:
                    self.detect(self.opt_settings['pipeline_settings']['vad_model'])
                case VoiceActivityDetectionTasks.SPLIT_BY_ACTIVITY:
                    self.split_audio()
        return

    def get_result(self):
        return self.latest_ret

