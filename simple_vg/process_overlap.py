from enum import Enum
from typing import Dict, TypedDict
from .commons import AbstractPipelineElement, ModelWrapper
from .speech_separation.sepreformer import SepReformerModels, SepReformerSetting, SepReformerWrapper
from .vg_types import AudioSetting
import copy


# https://www.isca-archive.org/interspeech_2023/yu23c_interspeech.pdf
# Detect Overlapped Speech
# Target Speech/Speaker Extraction => SpeakerBeam, but has multihead for separation for each speaker
# OSD(Overlapped Speech Detection) => ?, 
# Not Implemented Yet 
#
# I took https://github.com/desh2608/gss but it requires multichannel source or RTTM file which must contain timestamps for all speaker
# So, i tought that it cannot be used for general purpose since it requires human-tagging task(RTTM)
#
# https://github.com/dmlguq456/SepReformer/tree/main
# SOTA of Speech Separation


class ProcessOverlappedAudioTasks(Enum):
    SEPARATE_AUDIO = 0
    DETECT_OVERLAPPED = 1

class SeparateOverlappedModels(Enum):
    SepReformer = 0

class SeparateOverlappedSettings(TypedDict):
    model: SepReformerModels
    model_settings: SepReformerSetting

class DetectOverlappedSettings(TypedDict):
    temp: str

class ProcessOverlappedAudioSettings(TypedDict):
    tasks: ProcessOverlappedAudioTasks
    separation_settings: SeparateOverlappedSettings
    detection_settings : DetectOverlappedSettings


class ProcessOverlappedAudio(AbstractPipelineElement):
    def __init__(self, settings: AudioSetting):
        settings = copy.deepcopy(settings)
        self.settings: ProcessOverlappedAudioSettings = settings['opt_settings']
        self.audio_settings = settings
        return

    def _detect_overlapped(self):
        return
    
    def _separate_overlapped_audio(self):
        sep_name = self.settings['separation_settings']['model']
        match sep_name:
            case SeparateOverlappedModels.SepReformer:
                self._use_sepreformer()
            case _:
                raise KeyError(f'Unknown Settings: {sep_name}')
        return
    

    def _process_input(self, input):
        self.input = input
    
    def _execute(self):
        if self.settings['tasks'] == ProcessOverlappedAudioTasks.SEPARATE_AUDIO:
            self._separate_overlapped_audio()    
        
        if self.settings['tasks'] == ProcessOverlappedAudioTasks.DETECT_OVERLAPPED:
            self._detect_overlapped()
    
    def get_result(self):
        return self.result
    
    def _use_sepreformer(self):
        if self.settings['separation_settings']['model_settings']['sr'] is None:
            self.settings['separation_settings']['model_settings']['sr'] = self.audio_settings['sr']

        self.model = SepReformerWrapper(self.input, self.settings['separation_settings']['model_settings'])
        self.model.inference()
        self.result = self.model.get_result()
        return 


