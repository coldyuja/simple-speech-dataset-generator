# 아래 코드에서 사용할 Import 작성함.
import pathlib
from typing import *
from torch import Tensor
import os

# 아래 Option Class에서 Generic Type으로 사용할 TypeVar 정의함.
T = TypeVar('vg_T')

# 해당 패키지의 최상위 root path를 찾는 코드. 결과=<어떤 path>/simple_vg
# 패키지 내 특정 파일을 찾거나 import 할때 사용함.
ROOT_PATH = pathlib.Path(os.path.abspath(__file__)).parents[0]

# .common.AbstractPipelineElement를 정의하는데 사용한 Input, Output 타입임.
# 자세한 내용은 common.py에서 설명함.
In = TypeVar('PipelineInput')
Out = TypeVar('PipelineOutput')

# 해당 class를 이용시 dict의 key와 value의 type을 코드 작성 시 정확히 어떤 key,value를 사용할 수 있는지 명시하기 위해 작성함.
# 또한, 코드 작성시 TypedDict로 된 dict들은 특정 개발 도구에선 key나 value의 type을 힌트로 제공하기 때문에 코드 작성에 용이함.
# 해당 패키지 내 거의 모든 Setting들은 전부 TypedDict으로 정의되어 있음.
#
# AudioSetting Dict는 기본적으로 해당 패키지에서 사용할 때 필요한 설정들을 포함하고 있음.
# 모든 AbstractPipelineElement들은 AudioSetting을 초기화 시 입력으로 받고
# 각 Element의 특정한 설정은 opt_settings에 들어있음.
class AudioSetting(TypedDict):
    sr: int | None
    mono: bool | None
    n_channels: int | None
    n_fft: int | None
    hop_length: int | None
    window_size: int | None
    use_torch: bool = True
    opt_settings: Dict[Any, Any]

# 작성하게된 계기는 dict는 get으로 Optional한 결과를 얻는게 가능한데, list는 그것이 불가능하다.
# 따라서, Option class를 작성했음. Class의 이름이나 일부 method의 이름 방식은 Rust를 참고함.
# Type Hint를 위해 Generic 사용했음.
class Option(Generic[T]):
    # 초기화 시 값을 받아 None인지 아닌지 확인 후 저장함.
    def __init__(self, var: T):
        if var is None:
            self.is_none = True
        self.var = var

    # 해당 method 호출 시 만약 값이 None이라면 Exception 발생함
    def err_if_none(self):
        if self.is_none:
            raise ValueError('Option Error: value is None')
    
    # 해당 값이 None인지 아닌지 결과를 리턴함.
    def is_none(self) -> bool:
        return self.is_none
    
    # dict의 get과 동일한 결과를 리턴함. 값이 None이면 default값을 아니라면 해당 값을 리턴함. 
    def get(self, default: T|None=None) -> T | None:
        if self.is_none:
            return default
        else:
            return self.var

    # get과 달리 값이 None이라면 Exception 발생하고 아니라면 해당 값 리턴함.
    def unwrap_or(self, default: T|None=None) -> T:
        val = self.get(default=default)     
        self.err_if_none()
        return val

# 와일드카드로 import 할 경우에(from utils import *) import 될 object들을 __all__으로 정의할수 있고 정의함. 
__all__ = ['Option', 'AudioSetting']