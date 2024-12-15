# 해당 파일에서 사용할 모듈들을 Import함.
import copy
from enum import Enum
from typing import Dict, Iterable, List, Optional, TypedDict, Union
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
import whisper
import torch.nn.functional as F
import math
from simple_vg.utils import AttributeDummyClass, get_torch_device
from simple_vg.commons import ModelWrapper

# Whisper에서 사용가능한 모델 목록임.
class WhisperModels(Enum):
    WHISPER_TINY = 'tiny'
    WHISPER_TINY_EN = 'tiny.en'
    WHISPER_BASE = 'base'
    WHISPER_BASE_EN = 'base.en'
    WHISPER_SMALL = 'small'
    WHISPER_SMALL_EN = 'small.en'
    WHISPER_MEDIUM = 'medium'
    WHISPER_MEDIUM_EN = 'medium.en'
    WHISPER_LARGE = 'large'
    WHISPER_LARGE_V1 = 'large-v1'
    WHISPER_LARGE_V2 = 'large-v2'
    WHISPER_LARGE_V3 = 'large-v3'
    WHISPER_TURBO_V3 = 'large-v3-turbo'
    WHISPER_TURBO = 'turbo'

# Whisper가 수행할 수 있는 Task임.
# Task 옆에 있는 주석에 있는 X는 임의의 언어임.
class WhisperTasks(Enum):
    TRANSCRIBE = 'transcribe' # X -> X
    TRANSLATE = 'translate' # X -> EN

# whisper/whisper/decoding.py에서 코드 작성 편의를 위해 그대로 가져옴
class DecodingOptions(TypedDict):
    task: str
    language: Optional[str] = None
    temperature: float = 0.0
    sample_len: Optional[int] = None  # maximum number of tokens to sample
    best_of: Optional[int] = None  # number of independent sample trajectories, if t > 0
    beam_size: Optional[int] = None  # number of beams in beam search, if t == 0
    patience: Optional[float] = None  # patience in beam search (arxiv:2204.05424)
    length_penalty: Optional[float] = None
    prompt: Optional[Union[str, List[int]]] = None  # for the previous context
    prefix: Optional[Union[str, List[int]]] = None  # to prefix the current context
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    suppress_blank: bool = True  # this will suppress blank outputs
    without_timestamps: bool = False  # use <|notimestamps|> to sample text tokens only
    max_initial_timestamp: Optional[float] = 1.0
    fp16: bool = True  # use fp16 for most of the calculation

# WhisperWrapper의 설정임.
class WhisperSettings(TypedDict):
    decode_options: DecodingOptions
    model: WhisperModels
    task: WhisperTasks
    n_mels: int = None
    batch_size: int

# whisper/whisper/decoding.py에서 코드 작성 편의를 위해 그대로 가져옴
class DecodingResult(TypedDict):
    audio_features: Tensor
    language: str
    language_probs: Optional[Dict[str, float]] 
    tokens: List[int]
    text: str
    avg_logprob: float
    no_speech_prob: float 
    temperature: float
    compression_ratio: float

# Input을 가공하고 모델에 넣기 좋게 만들기 위한 WhisperWrapper에서 사용할 Dataset
# 아래의 코드 중 고정된 값들은 https://github.com/openai/whisper/blob/main/whisper/audio.py에서 확인 가능함.
class WhisperDataset(Dataset):
    def __init__(self, input: Tensor | list[Tensor], settings: WhisperSettings):
        self.raw_input = input
        self.settings = settings
        self.max_len = 3000 # Whisper에 입력가능한 최대 Sample 개수 (고정)
        self.indice = []

        if torch.is_tensor(self.raw_input):    
            # 입력이 Tensor인 경우 dimension이 2를 초과할 경우 강제로 2로 맞춤.
            # whisper.log_mel_spectrogram에 있는 torch.stft가 2D Tensor만 입력으로 받음.
            if len(self.raw_input.shape) > 2:
                s = self.raw_input.shape
                # 나중에 사용할 수도 있으니 Tensor의 직접적인 변형 말고 view를 사용함.
                self.raw_input = self.raw_input.view(math.prod(s[:-1]), s[-1])
            self.input = whisper.log_mel_spectrogram(self.raw_input, self.settings['n_mels'])
        elif isinstance(self.raw_input, (list)):
            if len(self.raw_input) == 0:
                raise ValueError('Input cannot empty!')

            if len(self.raw_input[0].shape) > 3:
                raise ValueError(f'Tensor elements in list must have 1 or 2-dimension. Current shape: {self.raw_input[0].shape}')
            self.input = [whisper.log_mel_spectrogram(inp, self.settings['n_mels']) for inp in self.raw_input]

        self._calc_indice_data()

    # 각 데이터를 3000 샘플씩 자르기 위한 Index를 구하는 method
    def _calc_indice_data(self):
        # self.input 의 최상위 dim의 길이를 data index로 삼는다.
        data_len = len(self.input)
        # 본래 다른 Audio Sample간 Slicing으로 인해 섞이면 안되므로
        # data index를 구하고 이를 slicing된 indice에 붙여 어느 데이터의 파트인지 구별
        for i in range(data_len):
            data_indice = self._calc_indice_single_tensor(self.input[i], i)
            self.indice += data_indice
        return
    
    # 여기선 단순히 3000 샘플씩 나누는 작업을 함.
    def _calc_indice_single_tensor(self, input, data_idx) -> list[list[int]]:
        max_len = self.max_len # 3000으로 고정
        # sample_len이 자르기 전 sample 개수임.
        sample_len = input.shape[-1]
        # ceil(sample_len/max_len)이 최종 indice의 개수이다.
        # range(int(sample_len/max_len))은 최종 indice의 개수 또는 그보다 1적은 범위에서 0부터 출력하는데
        # 그렇다면 ret에는 최종 indice의 개수 또는 1적은 개수가 들어갈 것이다.
        ret = [[data_idx, max_len * i, max_len * i+1] for i in range(int(sample_len/max_len))]

        # 따라서 ret의 가장 마지막 indice가 sample_len보다 작으면 해당 데이터 전체를 커버하지 않고 있으므로
        if len(ret) and ret[-1][2] < sample_len:
            # sample_len까지 포함하도록 indice를 추가한다.
            ret.append([data_idx, ret[-1][1], sample_len])
        # 해당 값이 0이면 원 데이터의 sample 수가 max_len=3000이 안되는 것이므로
        # 0부터 sample_len까지 추가한다.
        elif len(ret) == 0:
            ret.append([data_idx, 0, sample_len])
        return ret

    def __len__(self):
        return len(self.indice)
    
    # Dataset.__getitem__()은 모델에 넣을 한개의 데이터를 리턴한다.
    def __getitem__(self, index):
        data_i, sample_st, sample_ed = self.indice[index]
        return (data_i, self.input[data_i][..., sample_st:sample_ed])
    

class WhisperWrapper(ModelWrapper):
    def __init__(self, settings: WhisperSettings):
        self.settings = copy.deepcopy(settings)
        self.model = whisper.load_model(self.settings['model'].value)

        # 값이 비어있다면 기본값으로 전부 채움.
        self.settings['n_mels'] = self.model.dims.n_mels
        self.settings['decode_options']['task'] = self.settings['task'].value
        self.settings['decode_options'].setdefault('temperature', 0.0)
        self.settings['decode_options'].setdefault('suppress_tokens', '-1')
        self.settings['decode_options'].setdefault('suppress_blank', True)
        self.settings['decode_options'].setdefault('without_timestamps', False)
        self.settings['decode_options'].setdefault('max_initial_timestamp', 1.0)
        self.settings['decode_options'].setdefault('fp16', True)

        self.device = get_torch_device()

    # 입력이 들어오면 Dataset과 DataLoader 생성 후 저장
    def process_input(self, input: Tensor | list[Tensor]):
        self.dataset = WhisperDataset(input, self.settings)
        self.dataloader = DataLoader(self.dataset, self.settings['batch_size'], collate_fn=_collate)

    @torch.inference_mode
    def inference(self):
        ret_list = [[] for _ in range(len(self.dataset))]
        # AttributeDummyClass는 utils.py에서 설명했었음.
        # 간단히 dict을 attribute로도 값을 얻을수 있게 해주는 Class이다.
        decode_options = AttributeDummyClass(self.settings['decode_options'])
        # DataLoader에서 나오는 데이터는 batch_size까지 적용되어 나오는 데이터이다.
        for data_indice, mels in self.dataloader:
            mels = mels.to(self.device)
            ret = whisper.decode(self.model, mels, decode_options)
            for data_i, single_data in zip(data_indice, ret):
                if isinstance(single_data, (list)):
                    ret_list[data_i].append(*single_data)
                else:
                    ret_list[data_i].append(single_data)
        self.result = ret_list
        return
    
    def get_result(self) -> list[list[DecodingResult]]:

        return self.result

# DataLoader가 batch_size로 batch데이터를 만들때 사용하도록 제공하는 함수.
def _collate(inp):
    # Dataset.__getitem__()에서 [data_index, Tensor]가 나오는데
    # DataLoader를 거치면 [[data_index, Tensor]]가 된다.
    # 하지만 위에 있는 training loop에서는 DataLoader의 출력이 data_indice, Tensor가 되야 한다.
    # 따라서 이 함수를 적용하므로써 DataLoader의 기본출력을 원하는 형식으로 만들수 있다.
    data_i = [single_sample[0] for single_sample in inp]
    data = [single_sample[1] for single_sample in inp]

    # pad_sequence로 list[Tensor]을 하나의 Tensor로 만든다.
    data = pad_sequence(data, batch_first=True)
    data_len = data.shape[-1]

    # whisper.decode()는 무조건 sample 길이가 3000이 되야 하므로 0으로 padding 한다.
    if data_len < 3000:
        data = F.pad(data, (0,3000-data_len))
        
    return data_i, data
