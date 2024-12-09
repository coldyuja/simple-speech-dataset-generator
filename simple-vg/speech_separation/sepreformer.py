from typing import TypedDict

from torch import Tensor
from ..commons import ModelWrapper
from torch.utils.data import Dataset, DataLoader

class SepReformerSetting(TypedDict):
    sr: int
    chunk_max_len: int
    batch_size: int
    n_speaker: int = 2

class SepReformerDataset(Dataset):
    def __init__(self, input: Tensor, chunk_max_len: int):
        self.max_len = chunk_max_len
        self.input_data = input

    def __getitem__(self, index):
        

class SepReformerWrapper(ModelWrapper):
    # SepReformer is under Apache 2.0 License
    from ...SepReformer.models.SepReformer_Base_WSJ0 import main as sepreformer_b_wsj0
    from ...SepReformer.models.SepReformer_Large_DM_WHAM import main as sepreformer_l_dm_wham
    from ...SepReformer.models.SepReformer_Large_DM_WHAMR import main as sepreformer_l_dm_whamr
    from ...SepReformer.models.SepReformer_Large_DM_WSJ0 import main as sepreformer_l_dm_wsj0

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
    #
    # Object: librosa.load(path) | Tensor = (Tensor(n, )) -> slice(dur=max_len) -> inference -> concat() -> [sep_speech]


    def __init__(self, input, settings: SepReformerSetting):
        self.settings = settings
        
        return



    def _inference(self):
        return super()._inference()
    
    def _train(self):
        # Currently, training is not supported. (implement later)
        return super()._train()
    
    def get_result(self):
        return super().get_result()
