import copy
import torch
from torch import Tensor
from torchaudio import sox_effects
from .vg_types import AudioSetting
from typing import Generic, NewType, NoReturn, Any, TypedDict, AnyStr, Union
from .commons import AbstractPipelineElement
from enum import Enum

#Split by Voice Activity Detection

class Task(Enum):
    ACTIVITY_DETECTION = 1
    SPLIT_BY_ACTIVITY = 2

class VADPipelineSetting(TypedDict):
    vad_model: str | None

class VoiceActivityDetectionSetting(TypedDict):
    gain: int | None
    pitch: int | None
    effects: list[list[str]]
    tasks: list[Task]
    pipeline_settings: VADPipelineSetting | None

class Timestamp(TypedDict):
    start: int | float
    end: int | float

class TimestampUnit(Enum):
    SAMPLE = 1
    SEC = 2
    MSEC = 3


class VoiceActivityDetection(AbstractPipelineElement):
    def __init__(self, setting: AudioSetting):
        self.effects: list[list[str]] = []
        settings = copy.deepcopy(settings)
        self.settings = setting
        self.opt_settings: VoiceActivityDetectionSetting = setting['opt_settings']

        if setting['mono']:
            self.effects.append(['channels', '1'])
        if setting['sr']:
            self.effects.append(['rate', str(setting['sr'])])
        if setting['opt_settings']['gain']:
            self.effects.append(['gain', '-n', str(setting['opt_settings']['gain'])])
        if setting['opt_settings']['pitch']:
            self.effects.append(['pitch', str(setting['opt_settings']['pitch'])])
        if setting['opt_settings']['effects']:
            self.effects += setting['opt_settings']['effects']
        return
    
    def _process_input(self, input):
        # input_tensor shape: [channels, samples]
        if isinstance(input, (torch.Tensor)):
            self.input_tensor: Tensor = input
        elif isinstance(input, (AnyStr)):
            self.file_path = input
            self.input_tensor = None
    
    #Ref: silero_vad/utils_vad.py
    def _load_and_apply_effects(self) -> NoReturn:
        if self.input_tensor is None:
            wav, sr = sox_effects.apply_effects_file(self.file_path, self.effects)
        else:
            wav, sr = sox_effects.apply_effects_tensor(self.input_tensor, self.effects)

        self.data: Tensor = wav
        assert(self.settings['sr'] == sr)
        return
    
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
        wav = self.data.squeeze(0)
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero-vad')
        get_speech_timestamps, _, _, _, _ = utils
        self.timestamps: list[Timestamp] = get_speech_timestamps(wav, model)
        self.timestamp_unit = TimestampUnit.SAMPLE
        return
    
    def detect(self, model_name: str = "silero_vad") -> int:
        self._load_and_apply_effects()
        match model_name:
            case "silero_vad":
                self._use_silero_vad()
            case _:
                raise ValueError(f'Unknown model_name: {model_name}')

        self.latest_ret = self.timestamps
        return len(self.timestamps)

    # if you want to override timestamps, put new timestamps into below param.
    def split_audio(self, timestamps: list[Timestamp] = None, timestamp_unit: TimestampUnit=None) -> NoReturn:
        if timestamps is None:
            timestamps = self.timestamps
        if timestamp_unit is None:
            timestamp_unit = self.timestamp_unit

        self.splitted_audio = []

        for ts in timestamps:
            st, end = ts['start'], ts['end']


            # location of sample at t-sec = t * sr
            # Sampling Rate(sr) => the number of samples in 1 duration sec
            if timestamp_unit == TimestampUnit.SEC:
                st = st * self.sr
                end = end * self.sr
            elif timestamp_unit == TimestampUnit.MSEC:
                st = st * 1000 * self.sr
                end = end * 1000 * self.sr

            self.splitted_audio.append(self.data[st:end+1])    

        self.latest_ret = self.splitted_audio             
        return

    def _execute(self):
        for task in self.opt_settings['tasks']:
            match task:
                case Task.ACTIVITY_DETECTION:
                    self.detect(self.opt_settings['pipeline_settings']['vad_model'])
                case Task.SPLIT_BY_ACTIVITY:
                    self.split_audio()
        return

    def get_result(self):
        return self.latest_ret

