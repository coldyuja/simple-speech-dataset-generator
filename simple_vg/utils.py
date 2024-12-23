from typing import Any, NoReturn, TypeVar
import yaml
import torch
from .vg_types import *
import os
from os import PathLike
import sys

init_dir_names = [
        'raw_data',
        'audio',
        'processed',
        'temp',
]

init_dir_paths: list[str] = []
audio_extensions = {'mp3', 'wav', 'flac'}

T = TypeVar('T')

def get_elem(arr: list[T], idx=-1) -> Option[T]:
    arr_len = len(arr)
    if (idx >= 0 and arr_len <= idx) or (idx < 0 and arr_len < abs(idx)):
        return Option(None)
    else:
        return Option(arr[idx])

def create_init_dirs(root_dir: PathLike) -> NoReturn:
    for sub_name in init_dir_names:
        full_path = root_dir+sub_name
        init_dir_paths.append(full_path)       
        os.makedirs(full_path, exist_ok=True)
    return

def convert_to_audio(file_path: PathLike) -> NoReturn:
    basename = os.path.basename(file_path)
    names: list[str] = basename.split('.')
    ext = get_elem(names, 1)
    if not ext.is_none:
        ext = ext.get()
        if ext in audio_extensions:
            os.system(f'mv {file_path} {os.path.join(init_dir_names[1], basename)}')
            return
    
    os.system(
        f"ffmpeg -i {file_path} -f mp3 -vn -async 1 {os.path.join(init_dir_names[1], names[0]+'.mp3')}"
    )
    return

def get_torch_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
#From SepReformer/utils/util_system.py
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

def fill_dict(src: dict , dst: dict, target_keys=None):
    for key, value in src.items():
        if (target_keys and key in target_keys) or not target_keys:
            if not dst.get(key):
                dst[key] = value


class AttributeDummyClass:
    def __init__(self, d: dict):
        self.__dict__['inner'] = d

    def __setattr__(self, name, value):
        self.__dict__['inner'][name] = value

    def __getattr__(self, name):
        return self.__dict__['inner'].get(name)



