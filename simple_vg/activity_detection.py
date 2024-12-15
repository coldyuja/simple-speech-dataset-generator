# 해당 파일에서 사용할 모듈들을 Import함.
import copy
import torch
from torch import Tensor
from .vg_types import AudioSetting
from typing import NoReturn, TypedDict
from .commons import AbstractPipelineElement
from enum import Enum

# Voice Activity Detection 관련 수행 가능한 Task들 정의함.
class VoiceActivityDetectionTasks(Enum):
    ACTIVITY_DETECTION = 1
    SPLIT_BY_ACTIVITY = 2

# VAD(Voice Acvitivity Detection) 관련 사용가능 모델 목록
class VADModels(Enum):
    Silero_VAD = 0

# VoiceActivityDetection은 여러 Task를 한개의 Element로 실행할 수 있으므로
# 여러 Task에 대한 설정을 추가로 받음.
class VADPipelineSetting(TypedDict):
    vad_model: VADModels

# VAD에 관한 설정
class VoiceActivityDetectionSetting(TypedDict):
    tasks: list[VoiceActivityDetectionTasks]
    pipeline_settings: VADPipelineSetting | None

# 아래는 내부 출력 값의 type을 명시하기 위해 작성된 코드임.
class Timestamp(TypedDict):
    start: int | float
    end: int | float

class TimestampUnit(Enum):
    SAMPLE = 1
    SEC = 2
    MSEC = 3

# VAD는 대화 중 대화중이 아닐 때의 파트를 감지한다.
class VoiceActivityDetection(AbstractPipelineElement):
    def __init__(self, settings: AudioSetting):
        self.settings = copy.deepcopy(settings)
        self.opt_settings: VoiceActivityDetectionSetting = self.settings['opt_settings']

    # 입력: list[Tensor(C=1, S) or Tensor(S)] or Tensor(C=1, S) or Tensor(S)
    # 여기선 Batch가 있는 Tensor(B, C, S) 받지 않는데, 
    # Audio 별로 길이가 크게 다를 경우 한개의 Tensor임을 강조하게 되면 쓸데없는 Padding이 붙기 때문에 그렇다.
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
        # silero_vad use window_size=512 for 16KHz, 256 for 8KHz
        # silero_vad는 지원가능한 sampling rate가 있는데 그 sampling rate에 따른 window_size도 있다.
        # 아래는 이를 강제하는 과정이다.
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

        # torch.hub에 silero-vad가 올라가 있으니 이를 단순히 이용하면 된다.
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        get_speech_timestamps, _, _, _, _ = utils

        # 나중에 Type Infer로 코드 작성시 편하기 위해 Type 명시함.
        self.timestamps: list[list[Timestamp]] = []
        self.timestamp_units: list[TimestampUnit] = []

        for data in self.input_tensors:
            # get_speech_timestamps는 1D Tensor만 받는다. 이것 때문에 왜 Input Tensor Shape이 (C=1, S) or (S)인 이유이다.
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

        #Detection 결과는 self.timestamps에 저장된다.
        self.latest_ret = self.timestamps
        return len(self.timestamps)

    # detect()를 하지 않고 그냥 Audio Tensor를 나누기만 할 경우 또는 필요에 따라 timestamps와 timestamp_units를 덮어쓰기 할 수 있다.
    def split_audio(self, timestamps_list: list[list[Timestamp]] = None, timestamp_units: list[TimestampUnit]=None) -> NoReturn:
        if timestamps_list is None:
            timestamps_list = self.timestamps
        if timestamp_units is None:
            timestamp_units = self.timestamp_units

        self.splitted_audio = []

        # 단순히 Timestamp에 따라 Tensor을 Slicing한다.
        for i in range(len(timestamps_list)):
            curr_splitted_audio =  []
            for ts in timestamps_list[i]:
                st, end = ts['start'], ts['end']

                # t초 후 지나간 샘플 개수 = t * sr
                # Sampling Rate(sr) => 1초에 들어있는 샘플 개수
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

    # Task에 따라 분기하여 작업 실행
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

