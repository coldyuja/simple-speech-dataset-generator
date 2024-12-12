from enum import Enum
from typing import Any, TypedDict
import torch
from torch import Tensor, nn
from ..commons import ModelWrapper
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import importlib, pathlib
from vg_types import ROOT_PATH
import os
from os import PathLike
from utils import get_torch_device

class SepReformerModels(Enum):
    SepReformer_Base_WSJ0 = 0
    SepReformer_Large_DM_WHAM = 1
    SepReformer_Large_DM_WHAMR = 2
    SepReformer_Large_DM_WSJ0 = 3

class SepReformerSetting(TypedDict):
    sr: int
    chunk_max_len: int
    batch_size: int
    n_speaker: int = 2
    model: SepReformerModels
    distributed_gpu: bool = False

class SepReformerDataset(Dataset):
    def __init__(self, input: list[Tensor] | Tensor, chunk_max_len: int):
        # Input tensor shape: (B, C?, S)

        if isinstance(input, list):
            # Really does it need to check having multichannel?
            len_shape = len(input.shape)
            if len_shape == 2:
                self.mono = True
            elif len_shape == 3:
                self.mono = False
            else:
                raise KeyError(f"Unknown tensor shape: {input.shape}")

            input = pad_sequence(input, batch_first=True)
        
        self.max_len = chunk_max_len
        self.input_data: Tensor = input
        self.curr_data_idx = 0
        self.curr_sample_idx = 0
        self.sample_len = input.shape[-1]


    def __getitem__(self, index) -> Tensor:
        ret_data = None
        if self.max_len is None:
            ret_data = self.input_data[self.curr_data_idx]
            self.curr_data_idx += 1
        else:
            if self.max_len > len(self.input_data[self.curr_data_idx]) - self.curr_sample_idx+1:
                ret_data = self.input_data[self.curr_data_idx, ..., self.curr_sample_idx:]
                self.curr_sample_idx = 0
                self.curr_data_idx += 1
            else:
                next_sample_idx = self.curr_sample_idx + self.max_len
                ret_data = self.input_data[self.curr_data_idx, ..., self.curr_sample_idx:next_sample_idx]
                self.curr_sample_idx = next_sample_idx
        
        return ret_data
    

class SepReformerWrapper(ModelWrapper):
    # SepReformer is under Apache 2.0 License

    # Memo for wrapper of official SepReformer testing
    # Logging, calc losses, metrics and other codes or configs that not need to inference are excluded.
    # run.py
    # args =
    #   model: model names
    #   engine-mode: train, test, test_save, infer_sample
    #   sample-file: dir of sample audio
    #   out-wav-dir: (mode=test_save) dir of output wav
    # import_module models.{input: model}.main => main()
    #
    # models/{model}/main.py
    # func: main 
    # (codes or settings for wandb are ignored since i dont have any plan using wandb currently)
    # conf_yaml = load configuration yaml file and parse it
    # conf = conf_yaml['config']
    # dataloaders = get_dataloaders(args, conf['dataset'], conf['dataloader']) <- ./dataset.py
    # model: TorchModule = Model(**conf['model']) <- ./model.py
    # model.to(gpu)
    # crit, opt, sche = {each_factory} <- utils/util_implement.py
    # engine = Engine(all var of this func: main) <- ./engine.py
    # engine.run() => engine.Engine
    # 
    # engine.py
    # Engine.__init__()
    # [loss] = crit
    # main_opt = opt[0]
    # main_sch, warmup_sch = sch
    # chk_point = ./log/{scratch_weights|pretrained_weights}/*.{pth|.pt|<ext of torch checkpoint>}
    # 
    # Engine.run() => Engine._test()
    # 
    # torch.inference_mode()
    # input_sizes, mixture, src, key in dataloaders
    # pred = model(mixture)
    # 
    # dataloaders => dataset.py -> get_dataloaders() -> DataLoader -> collate_fn -> _collate([Datset.__getitem__()]) -> input_sizes, mixture, src, key ; but only mixture data needs.
    # _collate()
    # mixture <- pad_sequence(batch_first=True) <- d['mix'] <- d <- iter <- sorted(key=fn[x->x['num_sample']], rev=True) <- input=[Dataset.__getitem()]
    # ret Dataset.__getitem__ = {'num_sample': s_mix.shape[0], 'mix': s_mix, ...} <<- s_mix, _ <- Dataset._direct_load(key) <- key=Dataset.wave_keys[index] <- index <- <called by inference loop> 
    # Dataset.wave_keys = list(wave_dict_mix.keys())
    # ret Dataset._direct_load = s_mix <- ?s_mix[rand_idx(max_len)] <- s_mix, _ = librosa^lib.load(file, sr=fs) <- file=path <- wave_dict_mix[key] <- key
    # wave_dict_mix <- util_dataset_parse_scps() <- wave_scp_mix <- scp_config_mix <- conf['dataset']['mixture'] <- conf <- config.yaml
    #
    # max_len <- conf['dataset']['max_len'] <- Which basis used to derive max_len?
    # n(channel)?
    #
    # Object: librosa.load(path) | Tensor = (Tensor(n, )) -> slice(dur=max_len) | window? -> inference -> concat() -> [sep_speech]


    def __init__(self, input, settings: SepReformerSetting):
        self.settings = settings
        self.dataset = SepReformerDataset(input, settings['chunk_max_len'])
        self.dataloader = DataLoader(self.dataset, batch_size=settings['batch_size'], collate_fn=sepreformer_collate)
        self.device = get_torch_device()
        self.model = self._load_model_from_chkpoint()
        return

    def _load_config_yaml(self, yaml_path: PathLike):
        # From SepReformer/.../util_system.py
        from ...SepReformer.utils.util_system import parse_yaml
        return parse_yaml(yaml_path)

    def _load_model_from_chkpoint(self, custom_chk_path: PathLike = None):
        model_root_path = os.path.join(ROOT_PATH, f'SepReformer/models/{self.settings['model'].name}')
        yaml_path = os.path.join(model_root_path, 'configs.yaml')
        yaml_conf = self._load_config_yaml(yaml_path)
        config = yaml_conf['config']

        # From SepReformer/.../engine.py
        self.pretrain_weights_path = os.path.join(model_root_path, "log", "pretrain_weights")
        os.makedirs(self.pretrain_weights_path, exist_ok=True)
        self.scratch_weights_path = os.path.join(model_root_path, "log", "scratch_weights")
        os.makedirs(self.scratch_weights_path, exist_ok=True)
        self.checkpoint_path = self.pretrain_weights_path if any(file.endswith(('.pt', '.pt', '.pkl')) for file in os.listdir(self.pretrain_weights_path)) else self.scratch_weights_path
        
        model_mod = importlib.import_module(f'...SepReformer.models.{self.settings['model'].name}.model')
        model: nn.Module = model_mod.Model(**config['model'])
        checkpoint_dict = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint_dict['model_state_dict'], strict=False)
        model.to(self.device)
        return model

    def _test(self):

        return

    @torch.inference_mode
    def inference(self):
        self.model.eval()
        ret = []
        for data in self.dataloader:
            data.to(self.device)
            pred = self.model(data)
            print(pred.shape) # For Testing
            ret.append(pred)
        self.result = ret
        return
    
    def train(self):
        # Currently, training is not supported. (implement later)
        return super().train()
    
    def get_result(self) -> Any:
        return self.result


def sepreformer_collate(inp) -> Tensor:
    return pad_sequence(inp, batch_first=True)