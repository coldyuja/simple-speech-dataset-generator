import torch
from torchaudio import sox_effects
from vg_types import AudioSetting, PathLike
from typing import NoReturn, Any, TypedDict
 
#Split by Voice Activity Detection

class VoiceActivityDetectionSetting(TypedDict):
    gain: int | None
    pitch: int | None
    effects: list[list[str]]

class VoiceActivityDetection:
    def __init__(self, file_path: PathLike, setting: AudioSetting):
        self.file_path = file_path
        self.effects: list[list[str]] = []
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
    
    #Ref: silero_vad/utils_vad.py
    def _load_and_apply_effects(self) -> NoReturn:
        wav, sr = sox_effects.apply_effects_file(self.file_path, self.effects)
        self.data = wav
        assert(self.settings['sr'] == sr)
        return
    
    #silero-vad (MIT License)
    def _use_silero_vad(self) -> NoReturn:
        wav = self.data.squeeze(0)
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero-vad')
        get_speech_timestamps, _, _, _, _ = utils
        self.timestamps = get_speech_timestamps(wav, model, return_seconds=True)
        return
    
    def detect(self, model_name: str = "silero_vad") -> Any:
        self._load_and_apply_effects()
        match model_name:
            case "silero_vad":
                self._use_silero_vad()
            case _:
                raise ValueError(f'Unknown model_name: {model_name}')

        return self.timestamps


