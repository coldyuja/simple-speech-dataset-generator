# 해당 파일에서 사용할 모듈을 import함.
from enum import Enum
from typing import Any
from .commons import AbstractPipelineElement
from torch import Tensor
import torch
import numpy as np


# VGPipeline을 어떤 방식으로 실행할지 옵션을 명시함.
class ExecuteType(Enum):
    NoInit = 0 # VGPipeline을 그냥 초기화만 했을 경우 해당 값 사용됨.
    Sequential = 1 # Sequential하게 실행할 경우 해당 값 사용됨.

# VGPiepline을 정의함.
# 해당 class는 .common.AbstractPipelineElement으로 정의된 class들을 정해진 방식대로 실행함.
# 현재는 Sequential 방식으로만 실행 가능함. 추후 Functional 방식으로도 가능하게 구현 예정.
class VGPipeline:
    # 기본적인 초기화 코드. 그냥 초기화 한것 만으로는 동작하지 않는다.
    def __init__(self):
        self.run_type = ExecuteType.NoInit
        return
    
    # Sequential 로 실행할 경우 해당 method를 사용하여 추가적인 초기화 필요함.
    # 코드 자체는 단순히 class 내에 arg들을 저장하는 것 뿐임.
    def sequential(self, pipelines: list[type[AbstractPipelineElement]]):
        self.seq_pipes = pipelines
        self.run_type = ExecuteType.Sequential
        return
    
    # 초기화한대로 주어진 AbstractPipelineElement들을 실행함.
    # dbg, name arg들로 추가적인 정보를 출력하게 할 수 있음.
    def run(self, input: Any, dbg=False, name=None) -> Any:
        # 저장된 run_type: ExecuteType에 따라 실행함.
        # 현재 지원하는 ExecuteType은 Sequential 뿐이므로 해당 경우밖에 작동하지 않음.
        # 나머지는 KeyError 발생함.
        match self.run_type:
            case ExecuteType.Sequential:
                # _run_sequential method로 sequential하게 실행함.
                # 최종 출력은 class 내에 저장됨.
                self.result = self._run_sequential(input, dbg=dbg, name=name)
            case _:
                raise KeyError('Not Supported ExecutionType or VGPipeline is not initialized properly!')
            
        return self.result
    
    # sequential 하게 실행하는 원 method이다.
    # dbg, name으로는 디버깅 시 출력을 모니터링 하는데 도움을 준다.
    # cuda_empty_cache는 매 Element 실행이 끝날때 마다 torch.cuda.empty_cache()를 실행하여 나중에 실행될 코드에 영향을 최소화한다.
    def _run_sequential(self, input: Any, dbg=False, cuda_empty_cache=True, name=None) -> Any:
        mid_ret = input
        # torch에서 cuda가 사용 가능한지 확인함.
        cuda_enabled = torch.cuda.is_available()

        # dbg=Ture 일 경우 이름과 함께 정보를 출력하게 되는데 name이 None이면 그냥 빈 str로 대체하기 위해 작성함.
        if name:
            t_name = name+'\t' if name else ''

        # 여기에서 Element들을 실행한다.
        for i, p in enumerate(self.seq_pipes):
            p_name = p.__class__.__qualname__ # 디버깅 정보 출력시 사용하기 위해 class name 추출함.

            # Exception 발생시 해당 정보도 출력하기 위해 try-except문 사용함.
            try:
                # AbstractPipelineElement에 정의된 method들로 해당 Element 실행함.
                p._process_input(mid_ret)
                p._execute()
                # mid_ret으로 element 출력 저장후 다음 element에게 전달함.
                mid_ret = p.get_result()

                # dbg=True 일때 디버깅 정보를 출력하기 위한 코드임.
                if dbg:
                    # ValidateElement로 mid_ret의 정보를 출력하고자 했음.
                    # 정보량이 좀 한계가 있어 추후 해당 데이터가 어느 Element에서 나오고 어느 Element에게 들어갔는지 까지 출력하도록 수정 예정.
                    ve = ValidateElement()
                    mid_ret = ve._process(mid_ret)

                    # Pipeline의 이름과 Element Class 이름 그리고 출력의 Class name, 만약 Tensor라면 그 shape까지 출력하게 만들었음.
                    print(f"{t_name if name else ''}Index-{i} Element: {p_name} Output: {mid_ret.__class__.__qualname__}, shape: {mid_ret.shape if isinstance(mid_ret, (Tensor, np.ndarray)) else None}", flush=True)
            # Exception 발생 시 어느 Element에서 어느 오류가 발생했는지 출력하게 만들었음.
            except Exception as e:
                print(f"{t_name if name else ''}Exception Occurred during processing pipeline. \nIndex={i} \nElement={p_name}\n")
                raise e
            
            # 위에서 설명한 대로 cuda cache를 비운다.
            if cuda_enabled and cuda_empty_cache:
                torch.cuda.empty_cache()
        # 마지막 Element의 출력을 리턴한다.
        return mid_ret
    
# _run_sequential에서 설명한 대로 그냥 주어진 입력을 print로 출력하는 Class임.
# AbstractPipelineElement에 대한 자세한 설명은 해당 Class 코드에서 했음.
class ValidateElement(AbstractPipelineElement):
    def __init__(self):
        return
    
    # 여기서 해당 Class의 이름과 함께 주어진 메세지를 출력함
    def _print(self, msg):
        print(f'ValidateElement:\t{msg}', flush=True)

    # 그냥 입력받은 데이터를 그대로 출력함.
    def _process_input(self, input):
        self.input = input
        self._print(f'Data: {input}')
    
    def _execute(self):
        return 
    
    def get_result(self):
        return self.input
    