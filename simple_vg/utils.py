# 이 파일에서 사용할 모듈을 Import함.
from typing import Any, NoReturn, TypeVar
import yaml
import torch
from .vg_types import *
import os
from os import PathLike

# test.ipynb에선 미사용되었지만, Audio/Video를 동시 처리할 경우 사용될 디렉토리 이름들임.
init_dir_names = [
        'raw_data',
        'audio',
        'processed',
        'temp',
]

init_dir_paths: list[str] = []

# convert_to_audio에서 Audio파일인지 확인할 때 사용함.
audio_extensions = set('mp3', 'wav', 'flac')

T = TypeVar('T')

# vg_types.Option에서 설명했다시피 list의 element를 Optional하게 가져올수 있게 해준다.
# list.get(idx)가 안되서 작성한 코드임.
def get_elem(arr: list[T], idx=-1) -> Option[T]:
    arr_len = len(arr)
    if (idx >= 0 and arr_len <= idx) or (idx < 0 and arr_len < abs(idx)):
        return Option(None)
    else:
        return Option(arr[idx])

# 위에서 작성한 init_dir_names를 이용해 root_dir에 디렉토리를 생성함.
def create_init_dirs(root_dir: PathLike) -> NoReturn:
    for sub_name in init_dir_names:
        full_path = root_dir+sub_name
        init_dir_paths.append(full_path)       
        os.makedirs(full_path, exist_ok=True)
    return

# ffmpeg를 이용해 Videofile을 Audiofile로 변환해 저장함. 
def convert_to_audio(file_path: PathLike) -> NoReturn:
    basename = os.path.basename(file_path)
    names: list[str] = basename.split('.')
    ext = get_elem(names, 1)
    if not ext.is_none:
        ext = ext.get()

        # 이미 Audio인 경우 그대로 결과 디렉토리에 옮겨놓음.
        if ext in audio_extensions:
            os.system(f'mv {file_path} {os.path.join(init_dir_names[1], basename)}')
            return
    
    os.system(
        f"ffmpeg -i {file_path} -f mp3 -vn -async 1 {os.path.join(init_dir_names[1], names[0]+'.mp3')}"
    )
    return

# cuda -> cpu 순으로 torch.device를 불러와 이를 리턴해줌.
# 이 패키지 내 많은 곳에서 사용됨.
def get_torch_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
# SepReformer/utils/util_system.py에서 그대로 가져옴
# yaml 파일을 불러오는 코드임.
def parse_yaml(path):
    """
    Parse and return the contents of a YAML file.

    Args:
        path (str): Path to the YAML file to be parsed.

    Returns:
        dict: A dictionary containing the parsed contents of the YAML file.

    Raises:
        FileNotFoundError: If the provided path does not point to an existing file.
    """
    try:
        with open(path, 'r') as yaml_file:
            config_dict = yaml.full_load(yaml_file)
        return config_dict
    except FileNotFoundError:
        raise

# Setting값을 상속시킬때 사용함.
# src에 있는 key, value를 dst에 값이 있으면 넣고 아니면 마는 방식임.
def fill_dict(src: dict , dst: dict, target_keys=None):
    for key, value in src.items():
        if (target_keys and key in target_keys) or not target_keys:
            dst.setdefault(key, value)


# WhisperWrapper에서 사용하는데 
# 원래 Setting들은 전부 TypedDict으로 dict type임.
# 하지만 Whisper의 DecodingOptions은 dataclass고 attribute로 값에 접근할수 있음.
# 그런데 dict는 이게 지원되지 않으므로 따로 이가 가능하게 Wrapping하는 Class가 필요해 이를 생성했음.
# .stt.whisper_wrapper에서 사용 예를 확인 가능함.
class AttributeDummyClass:
    # 내부적으로는 주어진 dict을 저장했다가 attribute로 접근시 해당 dict에 있는 값을 리턴하고
    # attribute를 set할 경우 마찬가지로 dict에 저장함.
    def __init__(self, d: dict):
        self.__dict__['inner'] = d

    def __setattr__(self, name, value):
        self.__dict__['inner'][name] = value

    def __getattr__(self, name):
        return self.__dict__['inner'].get(name)



