import copy
from enum import Enum
import sys
from typing import Any, TypedDict
import torch
from torch import Tensor, nn
from ..commons import ModelWrapper
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import importlib, pathlib
from ..vg_types import ROOT_PATH
import os
from os import PathLike
from ..utils import get_torch_device, parse_yaml


class SepReformerModels(Enum):
    SepReformer_Base_WSJ0 = 0
    SepReformer_Large_DM_WHAM = 1
    SepReformer_Large_DM_WHAMR = 2
    SepReformer_Large_DM_WSJ0 = 3

class SepReformerSetting(TypedDict):
    sr: int
    chunk_max_len: int
    batch_size: int
    n_speaker: int # only n_speaker=2 tested on paper. 
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

        if self.max_len is None:
            self.max_len = sys.maxsize

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

class SepReformerWrapper(ModelWrapper):
    # SepReformer is under Apache 2.0 License

    def __init__(self, input, settings: SepReformerSetting):
        settings = copy.deepcopy(settings)
        self.settings = settings
        if settings['distributed_gpu']:
            print("Warning: Distributed GPU is not currenly supported. Only single gpu will be used.")

        self.dataset = SepReformerDataset(input, settings['chunk_max_len'])
        self.dataloader = DataLoader(self.dataset, batch_size=settings['batch_size'], collate_fn=sepreformer_collate)
        self.device = get_torch_device()
        self.model = self._load_model_from_chkpoint()
        return

    def _load_config_yaml(self, yaml_path: PathLike):
        return parse_yaml(yaml_path)

    def _load_model_from_chkpoint(self, custom_chk_path: PathLike = None):
        model_root_path = os.path.join(ROOT_PATH, f'SepReformer/models/{self.settings["model"].name}')
        yaml_path = os.path.join(model_root_path, 'configs.yaml')
        yaml_conf = self._load_config_yaml(yaml_path)
        config = yaml_conf['config']

        # From SepReformer/.../engine.py
        self.pretrain_weights_path = os.path.join(model_root_path, "log", "pretrain_weights")
        os.makedirs(self.pretrain_weights_path, exist_ok=True)
        self.scratch_weights_path = os.path.join(model_root_path, "log", "scratch_weights")
        os.makedirs(self.scratch_weights_path, exist_ok=True)
        self.checkpoint_path = self.pretrain_weights_path if any(file.endswith(('.pt', '.pt', '.pkl')) for file in os.listdir(self.pretrain_weights_path)) else self.scratch_weights_path

        chkpoint_list = sorted([filename for filename in os.listdir(self.checkpoint_path)], key=lambda name: int(name.split('.')[1]))
        self.checkpoint_path = os.path.join(self.checkpoint_path, chkpoint_list[-1])
    
        model_mod = importlib.import_module(f'.models.{self.settings["model"].name}.model', 'simple_vg.speech_separation')
        config['model']['num_spks'] = self.settings['n_speaker']

        model: nn.Module = model_mod.Model(**config['model'])
        checkpoint_dict = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint_dict['model_state_dict'], strict=False)
        model = model.to(self.device)
        return model

    def _test(self):

        return

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


def sepreformer_collate(inp) -> Tensor:
    return pad_sequence(inp, batch_first=True)

