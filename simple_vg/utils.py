from typing import Any, Never, NoReturn, TypeVar

import torch
from vg_types import *
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
        f'ffmpeg -i {file_path} -f mp3 -vn -async 1 {os.path.join(init_dir_names[1], names[0]+'.mp3')}'
    )
    return

def get_torch_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")