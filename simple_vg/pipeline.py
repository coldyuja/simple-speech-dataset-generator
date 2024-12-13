from enum import Enum
from typing import Any
from .commons import AbstractPipelineElement
from torch import Tensor
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

    def run(self, input: Any, dbg=False):
        match self.run_type:
            case ExecuteType.Sequential:
                self._run_sequential(input, dbg=dbg)
            case _:
                raise KeyError('Not Supported ExecutionType or VGPipeline is not initialized properly!')
            
        return
    
    # I wanted to give type hints more detail, but [T] = Generic[T] does not supported python 3.10
    # If i use Generic[T], code will be messy. so that's why i didnt type hinting on this method
    def _run_sequential(self, input: Any, dbg=False) -> Any:
        mid_ret = input
        for i, p in enumerate(self.seq_pipes):
            p_name = p.__class__.__qualname__
            try:
                p._process_input(mid_ret)
                p._execute()
                mid_ret = p.get_result()
                if dbg:
                    print(f'Index-{i} {p_name} Pipeline Output: {mid_ret.__class__.__qualname__}, shape: {mid_ret.shape if isinstance(mid_ret, (Tensor, np.ndarray)) else None}')
            except Exception as e:
                print(f'Exception Occurred during processing pipeline. \nIndex={i} \nPipeline={p_name}\n')
                raise e
        return mid_ret
    
