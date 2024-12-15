# 해당 파일에서 사용할 모듈들을 Import함.
import copy
from enum import Enum
import sys
from typing import Any, TypedDict
import torch
from torch import Tensor, nn
from ..commons import ModelWrapper
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import importlib
from ..vg_types import ROOT_PATH
import os
from os import PathLike
from ..utils import get_torch_device, parse_yaml

# SepReformer에서 존재하는 모델임
# Base_WSJ0 제외하고는 현재 저자가 공개를 하지 않았으므로 0번만 사용 가능.
class SepReformerModels(Enum):
    SepReformer_Base_WSJ0 = 0
    SepReformer_Large_DM_WHAM = 1
    SepReformer_Large_DM_WHAMR = 2
    SepReformer_Large_DM_WSJ0 = 3

# SepReformer 설정 가능 목록
class SepReformerSetting(TypedDict):
    sr: int
    chunk_max_len: int
    batch_size: int
    n_speaker: int # n_speaker=2인 경우만 저자가 검증했음.
    model: SepReformerModels
    distributed_gpu: bool = False

class SepReformerDataset(Dataset):
    def __init__(self, input: list[Tensor] | Tensor, chunk_max_len: int):
        # Input tensor shape: (B, C?, S)

        input = pad_sequence(input, batch_first=True)
        if len(input.shape) == 2:
            input = torch.unsqueeze(input, 1)
        
        self.max_len = chunk_max_len
        self.input_data: Tensor = input
        self.curr_data_idx = 0
        self.curr_sample_idx = 0
        self.sample_len = input.shape[-1]

        self.sample_indice = []
        self.data_indice = []

        self.data_len = input.shape[0]
        self.sample_len = input.shape[-1]

        # 따로 최대값이 지정되지 않는 경우 int의 최대값으로 한다.
        if self.max_len is None:
            self.max_len = sys.maxsize

        # self.sample_indice에는 sample의 range가 담기고
        # self.data_indice에는 data의 index가 담긴다.
        # 두 list의 길이는 동일하며 한개의 index로 data_index, sample_start, sample_end 을 얻을 수 있다.
        for d_i in range(self.data_len):
            s_i = 0
            while True:
                self.data_indice.append(d_i)
                if self.sample_len - s_i+1 <= self.max_len:
                    self.sample_indice.append((s_i, self.sample_len))
                    break
                else:
                    self.sample_indice.append((s_i, s_i + self.max_len))

    def __len__(self):
        return len(self.sample_indice)


    def __getitem__(self, index) -> Tensor:
        data_i = self.data_indice[index]
        sample_st, sample_ed = self.sample_indice[index]
        ret = self.input_data[data_i, ..., sample_st:sample_ed]
        ret = torch.squeeze(ret, dim=0)
        return ret

# SepReformer의 Wrapper
class SepReformerWrapper(ModelWrapper):
    # SepReformer은 Apache 2.0 License 적용됨.

    def __init__(self, input, settings: SepReformerSetting):
        settings = copy.deepcopy(settings)
        self.settings = settings
        # 현재 distributed gpu는 구현되지 않음.
        if settings['distributed_gpu']:
            print("Warning: Distributed GPU is not currenly supported. Only single gpu will be used.")

        # 주어진 입력으로 Dataset과 DataLoader 저장
        self.dataset = SepReformerDataset(input, settings['chunk_max_len'])
        self.dataloader = DataLoader(self.dataset, batch_size=settings['batch_size'], collate_fn=sepreformer_collate)
        self.device = get_torch_device()
        self.model = self._load_model_from_chkpoint()
        return

    def _load_config_yaml(self, yaml_path: PathLike):
        return parse_yaml(yaml_path)

    def _load_model_from_chkpoint(self, custom_chk_path: PathLike = None):
        model_root_path = os.path.join(ROOT_PATH, f'SepReformer/models/{self.settings["model"].name}')
        # 모델 설정 불러옴
        yaml_path = os.path.join(model_root_path, 'configs.yaml')
        yaml_conf = self._load_config_yaml(yaml_path)
        config = yaml_conf['config']

        # SepReformer/.../engine.py에서 그대로 가져옴
        self.pretrain_weights_path = os.path.join(model_root_path, "log", "pretrain_weights")
        os.makedirs(self.pretrain_weights_path, exist_ok=True)
        self.scratch_weights_path = os.path.join(model_root_path, "log", "scratch_weights")
        os.makedirs(self.scratch_weights_path, exist_ok=True)
        self.checkpoint_path = self.pretrain_weights_path if any(file.endswith(('.pt', '.pt', '.pkl')) for file in os.listdir(self.pretrain_weights_path)) else self.scratch_weights_path

        chkpoint_list = sorted([filename for filename in os.listdir(self.checkpoint_path)], key=lambda name: int(name.split('.')[1]))
        self.checkpoint_path = os.path.join(self.checkpoint_path, chkpoint_list[-1])

        # 위에서 말했듯이 이 모델 말고는 아직 공개되지 않았음.
        if self.settings['model'] != SepReformerModels.SepReformer_Base_WSJ0:
            raise NotImplementedError('This Model is not published by author currently. Only SepReformer-B can be used.')

        # SepReformer은 model 별로 패키지가 나뉘어져 있는 방식이기 때문에 따로 import 해줘야됨.
        model_mod = importlib.import_module(f'.models.{self.settings["model"].name}.model', 'simple_vg.speech_separation')
        config['model']['num_spks'] = self.settings['n_speaker']

        # 모델을 로드하고
        model: nn.Module = model_mod.Model(**config['model'])
        # pretrained 가중치를 가져오고
        checkpoint_dict = torch.load(self.checkpoint_path, map_location=self.device)
        # 그 모델에 적용함.
        model.load_state_dict(checkpoint_dict['model_state_dict'], strict=False)
        model = model.to(self.device)
        return model

    def process_input(self, input):
        return

    def _test(self):

        return
    # torch.inference_mode는 inference만 할때 더 처리를 빠르게 해줌
    @torch.inference_mode
    def inference(self):
        self.model.eval()
        ret = []
        for data in self.dataloader:
            data = data.to(self.device)
            pred, _ = self.model(data)
            pred = pad_sequence(pred, batch_first=True)
            ret.append(pred)
        self.result = pad_sequence(ret, batch_first=True)
        return
    
    def train(self):
        # Currently, training is not supported. (implement later)
        return super().train()
    
    def get_result(self) -> Any:
        return self.result

# DataLoader에서 사용함.
# 단순히 list[Tensor]을 한개의 Tensor로 변형하기 위한것.
def sepreformer_collate(inp) -> Tensor:
    return pad_sequence(inp, batch_first=True)

