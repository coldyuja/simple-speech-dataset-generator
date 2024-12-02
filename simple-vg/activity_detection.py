import torch
from torchaudio import sox_effects
from vg_types import PathLike
from typing import NoReturn, Any
 
#Split by Voice Activity Detection

class VoiceActivityDetection:
    def __init__(self, file_path: PathLike,
                 mono=True,
                 sr=16000,
                 gain=None,
                 pitch=None,
                 effects: list[list[str]] | None = None):
        self.file_path = file_path
        self.effects: list[list[str]] = []

        if mono:
            self.effects.append(['channels', '1'])
        if sr:
            self.effects.append(['rate', str(sr)])
        if gain:
            self.effects.append(['gain', '-n', str(gain)])
        if pitch:
            self.effects.append(['pitch', str(pitch)])
        if effects:
            self.effects += effects

        return
    
    #Ref: silero_vad/utils_vad.py
    def _load_and_apply_effects(self) -> NoReturn:
        wav, sr = sox_effects.apply_effects_file(self.file_path, self.effects)
        self.data = wav
        self.ret_sr = sr
        return
    
    #silero-vad (MIT License)
    def _use_silero_vad(self) -> NoReturn:
        wav = self.data.squeeze(0)
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero-vad')
        get_speech_timestamps, _, _, _, _ = utils
        self.timestamps = get_speech_timestamps(wav, model, return_seconds=True)
        return
    
    def detect(self, model_name: str) -> Any:
        self._load_and_apply_effects()
        match model_name:
            case "silero_vad":
                self._use_silero_vad()
            case _:
                raise ValueError(f'Unknown model_name: {model_name}')

        return self.timestamps


