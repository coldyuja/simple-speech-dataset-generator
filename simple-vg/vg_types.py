from typing import *
from torch import Tensor

T = TypeVar('vg_T')
PathLike = TypeVar('PathLike', str, Any)
AudioData = TypeVar('AudioData', bytes)
TensorOrPath = TypeVar('TensorOrPath', AnyStr, Tensor)



In = TypeVar('PipelineInput')
Out = TypeVar('PipelineOutput')

class AudioSetting(TypedDict):
    sr: int | None
    mono: bool | None
    n_fft: int | None
    window_size: int | None
    use_torch: bool = True
    opt_settings: Dict[Any, Any]


class Option(Generic[T]):
    def __init__(self, var: T):
        if var is None:
            self.is_none = True
        self.var = var

    def err_if_none(self):
        if self.is_none:
            raise ValueError('Option Error: value is None')
        
    def is_none(self) -> bool:
        return self.is_none
    
    def get(self, default: T|None=None) -> T | None:
        if self.is_none:
            return default
        else:
            return self.var

    def unwrap_or(self, default: T|None=None) -> T:
        val = self.get(default=default)
        if val is None:
            self.err_if_none()
        return val
    
    

class SafeList(List):
    def __init__(self):
        super().__init__()




__all__ = ['PathLike', 'AudioData', 'Option', 'AudioSetting']