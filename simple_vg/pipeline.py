from enum import Enum
from typing import Any

from .commons import AbstractPipelineElement
from torch import Tensor
import torch
import numpy as np

class ExecuteType(Enum):
    NoInit = 0
    Sequential = 1

class VGPipeline:
    def __init__(self):
        self.run_type = ExecuteType.NoInit
        return
    
    def sequential(self, pipelines: list[type[AbstractPipelineElement]]):
        self.seq_pipes = pipelines
        self.run_type = ExecuteType.Sequential
        return
    
    def run(self, input: Any, dbg=False, name=None) -> Any:
        match self.run_type:
            case ExecuteType.Sequential:
                self.result = self._run_sequential(input, dbg=dbg, name=name)
            case _:
                raise KeyError('Not Supported ExecutionType or VGPipeline is not initialized properly!')
            
        return self.result
    
    # I wanted to give type hints more detail, but [T] = Generic[T] does not supported python 3.10
    # If i use Generic[T], code will be messy. so that's why i didnt type hinting on this method
    def _run_sequential(self, input: Any, dbg=False, cuda_empty_cache=True, name=None) -> Any:
        mid_ret = input
        cuda_enabled = torch.cuda.is_available()
        if name:
            t_name = name+'\t' if name else ''

        for i, p in enumerate(self.seq_pipes):
            p_name = p.__class__.__qualname__
            try:
                p._process_input(mid_ret)
                p._execute()
                mid_ret = p.get_result()
                if dbg:
                    ve = ValidateElement()
                    mid_ret = ve._process(mid_ret)
                    print(f"{t_name if name else ''}Index-{i} Element: {p_name} Output: {mid_ret.__class__.__qualname__}, shape: {mid_ret.shape if isinstance(mid_ret, (Tensor, np.ndarray)) else None}", flush=True)
            except Exception as e:
                print(f"{t_name if name else ''}Exception Occurred during processing pipeline. \nIndex={i} \nElement={p_name}\n")
                raise e
            if cuda_enabled and cuda_empty_cache:
                torch.cuda.empty_cache()
        return mid_ret
    

class ValidateElement(AbstractPipelineElement):
    def __init__(self):
        return
    
    def _print(self, msg):
        print(f'ValidateElement:\t{msg}', flush=True)

    def _process_input(self, input):
        self.input = input
        self._print(f'Data: {input}')
    
    def _execute(self):
        return 
    
    def get_result(self):
        return self.input
    