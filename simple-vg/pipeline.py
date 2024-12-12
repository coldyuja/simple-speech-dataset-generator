from enum import Enum
from typing import Any
from commons import AbstractPipelineElement

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

    def run(self, input: Any):
        match self.run_type:
            case ExecuteType.Sequential:
                self._run_sequential(input)
            case _:
                raise KeyError('Not Supported ExecutionType or VGPipeline is not initialized properly!')
            
        return
    
    # I wanted to give type hints more detail, but [T] = Generic[T] does not supported python 3.10
    # If i use Generic[T], code will be messy. so that's why i didnt this type hinting
    def _run_sequential(self, input: Any) -> Any:
        mid_ret = None
        for i, p in enumerate(self.seq_pipes):
            try:
                p._process_input(input)
                p._execute()
                mid_ret = p.get_result()
            except Exception as e:
                print(f'Exception Occurred during processing pipeline. \nIndex={i} \nPipeline={p.__name__} \nError:\n{e}\n')

        return mid_ret