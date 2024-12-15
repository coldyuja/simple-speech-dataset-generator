# 해당 파일에서 사용할 모듈들을 Import함.
import copy
from enum import Enum
from typing import Any, TypedDict
from .commons import AbstractPipelineElement
from .vg_types import AudioSetting
from .stt.whisper_wrapper import WhisperSettings, WhisperWrapper, WhisperTasks, DecodingOptions
from .utils import fill_dict

# 해당 패키지에서는 이런 모델을 단순히 Wrapping해서 제공하는 것이므로
# Whisper에 대한 자세한 설명은 https://github.com/openai/whisper를 참고바람.

# Whisper의 사용가능한 모든 모델 명칭을 가져왔음.
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

# Speech-to-Text 작업시 사용할 설정들이 명시되어 있음.
class ExtractTextSettings(TypedDict):
    model: ExtractTextModels
    verbose: bool
    batch_size: int
    model_settings: Any
    max_chunk_len: int | None

# Speech-to-Text를 수행하는 Element
class ExtractText(AbstractPipelineElement):
    def __init__(self, settings: AudioSetting):
        # 변경 방지를 위해 setting을 deepcopy함.
        self.settings = copy.deepcopy(settings)
        self.model_settings: ExtractTextSettings = self.settings['opt_settings']
        
        # Element 설정에 있는 값을 모델 설정에 없을 경우 적용함.
        fill_dict(self.model_settings, self.model_settings['model_settings'], set(['batch_size', 'model']))

        # 모델 초기화시 시간이 일부 소요될 수 있으므로 초기화시 같이 미리 초기화함.
        if 'WHISPER' in self.model_settings['model'].name:
            self.model = WhisperWrapper(self.model_settings['model_settings'])
        return
    
    def _process_input(self, input):
        # Input Tensor shape: (B, C, S) or list[(C, S)] 
        self.input = input

    # 초기화된 모델에 넣고 출력을 받아 그대로 최종 리턴값으로 출력함.
    def _execute(self):
        self.model.process_input(self.input)
        self.model.inference()
        self.result = self.model.get_result()
        return
    
    def get_result(self):
        return self.result
    
# ExtractTextSettings의 기본 설정값을 얻을 수 있음.
def default_settings() -> ExtractTextSettings:
    w_settings: WhisperSettings = {
        'task': WhisperTasks.TRANSCRIBE,
        'decode_options': DecodingOptions()
    }
    settings: ExtractTextSettings = {
        'batch_size': 1,
        'max_chunk_len': None,
        # 성능은 좋으면서 Inference가 동일 성능대비 좋으므로 선택함.
        'model': ExtractTextModels.WHISPER_TURBO,
        'model_settings': w_settings,
    }

    return settings

    