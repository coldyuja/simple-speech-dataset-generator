# 해당 파일에서 사용할 모듈들을 Import함.
from enum import Enum
from typing import Dict, TypedDict
from .commons import AbstractPipelineElement, ModelWrapper
from .speech_separation.sepreformer import SepReformerModels, SepReformerSetting, SepReformerWrapper
from .vg_types import AudioSetting
import copy

# 여러 대화가 겹쳐진 Audio에 대한 Task들을 정의했음.
class ProcessOverlappedAudioTasks(Enum):
    SEPARATE_AUDIO = 0 # Whisper을 사용해 가능하게 했음
    DETECT_OVERLAPPED = 1 # 아직 미구현 상태임. 나중을 위해 이름만 추가해 놓았음.

# 겹쳐진 Audio를 분리시키기 위해 사용할 모델들을 정의함.
class SeparateOverlappedModels(Enum):
    SepReformer = 0

# Sepreformer를 설정하기 위한 Settings dict임.
class SeparateOverlappedSettings(TypedDict):
    model: SepReformerModels # 여기선 모델을 명시함.
    model_settings: SepReformerSetting # 여기선 SepReformer 설정을 명시함.

# 미구현된 Task의 임시 Setting
class DetectOverlappedSettings(TypedDict):
    temp: str

# 여기선 수행할 Task와 각 Task의 설정을 가짐.
class ProcessOverlappedAudioSettings(TypedDict):
    tasks: ProcessOverlappedAudioTasks
    separation_settings: SeparateOverlappedSettings
    detection_settings : DetectOverlappedSettings

# 겹쳐진 Speech에 대한 작업을 수행하는 Element
class ProcessOverlappedAudio(AbstractPipelineElement):
    def __init__(self, settings: AudioSetting):
        # 타 코드에 의해 변경 방지를 위해 deepcopy 수행함.
        settings = copy.deepcopy(settings)
        self.settings: ProcessOverlappedAudioSettings = settings['opt_settings']
        self.audio_settings = settings
        return

    # 미 구현된 기능임. 나중 버전에서 구현 예정.
    def _detect_overlapped(self):
        raise NotImplementedError("Overlap Detection is not implemented yet.")
        
    # 선택된 모델에 따라 분기 후 코드 실행됨.
    def _separate_overlapped_audio(self):
        sep_name = self.settings['separation_settings']['model']
        match sep_name:
            case SeparateOverlappedModels.SepReformer:
                self._use_sepreformer()
            case _:
                raise KeyError(f'Unknown Settings: {sep_name}')
        return
    
    # 입력: 각 Task의 각 Model별로 다름.
    def _process_input(self, input):
        self.input = input
    
    # 선택된 Task에 따라 분기 후 코드 실행됨.
    def _execute(self):
        if self.settings['tasks'] == ProcessOverlappedAudioTasks.SEPARATE_AUDIO:
            self._separate_overlapped_audio()    
        
        if self.settings['tasks'] == ProcessOverlappedAudioTasks.DETECT_OVERLAPPED:
            self._detect_overlapped()
    
    # 출력: 각 Task의 각 Model별로 다름.
    def get_result(self):
        return self.result
    
    # 저장된 특정 Task에 대한 설정들과 함께 ModelWrapper에 넣고 결과를 얻게됨.
    def _use_sepreformer(self):
        if self.settings['separation_settings']['model_settings']['sr'] is None:
            self.settings['separation_settings']['model_settings']['sr'] = self.audio_settings['sr']

        # SepReformerWrapper는 ModelWrapper로 이는 .common.ModelWrapper에서 설명함.
        self.model = SepReformerWrapper(self.input, self.settings['separation_settings']['model_settings'])
        self.model.inference()
        self.result = self.model.get_result()
        return 


