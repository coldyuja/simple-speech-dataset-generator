from typing import Any, Callable, TypeVar

import torch

from simple_vg.pipeline import VGPipeline
from .commons import AbstractPipelineElement
from torch import Tensor
import numpy as np

input_t = TypeVar('Input')
output_t = TypeVar('Output')
input_fn_t = Callable[[type[AbstractPipelineElement], input_t], Any]
exec_fn_t = Callable[[type[AbstractPipelineElement], Any], output_t]

# Not the best name..
class ClosureElement(AbstractPipelineElement):
    def __init__(self, input_fn: input_fn_t, exec_fn: exec_fn_t, opt_data=None):
        self.input_fn = input_fn
        self.exec_fn = exec_fn
        self.opt_data = opt_data

    def _process_input(self, input):
        if self.input_fn:
            self.input = self.input_fn(self, input)
        else:
            self.input = input

    def _execute(self):
        if self.exec_fn:
            self.result = self.exec_fn(self, self.input)
        else:
            self.result = self.input
    
    def get_result(self):
        return self.result
    
class ParallelElement(AbstractPipelineElement):
    def __init__(self, pipelines: list[list[type[AbstractPipelineElement]]], dbg=False, name: str | None=None):
        self.pipelines = pipelines
        self.dbg = dbg
        self.name = name if name else self.__class__.__name__
        self.p_list = []
    def _print(self, msg):
        print(f'{self.name}\t{msg}', flush=True)

    def _process_input(self, input):
        self.root_input = input
        if self.dbg:
            self._print(f'Root Input: {self.root_input.__class__.__qualname__}, shape: {self.root_input.shape if isinstance(self.root_input, (Tensor, np.ndarray)) else None}')
    
    def _execute(self):
        results = []
        for i, elem_list in enumerate(self.pipelines):
            p = VGPipeline()
            p.sequential(elem_list)
            p_name = self.name + f'-{i}'
            sub_result = p.run(self.root_input, dbg=self.dbg, name=p_name)
            self.p_list.append(p)
            results.append(sub_result)
        
        self.result = results
    
    def get_result(self):
        return self.result
    
class SequentialElement(AbstractPipelineElement):
    def __init__(self, pipelines: list[type[AbstractPipelineElement]], dbg=False, name: str | None=None):
        self.pipelines = pipelines
        self.dbg = dbg
        self.name = name if name else self.__class__.__name__

    def _print(self, msg):
        print(f'{self.name}\t{msg}', flush=True)

    def _process_input(self, input):
        self.root_input = input
        if self.dbg:
            self._print(f'Root Input: {self.root_input.__class__.__qualname__}, shape: {self.root_input.shape if isinstance(self.root_input, (Tensor, np.ndarray)) else None}')
    
    def _execute(self):
        p = VGPipeline()
        p.sequential(self.pipelines)
        p_name = self.name
        result = p.run(self.root_input, dbg=self.dbg, name=p_name)
        self.pipe = p

        self.result = result
    
    def get_result(self):
        return self.result
    

class TransparentElement(AbstractPipelineElement):
    def __init__(self):
        return
    
    def _process_input(self, input):
        self.input = input
        return
    
    def _execute(self):
        return
    
    def get_result(self):
        return self.input
    

class InputElement(AbstractPipelineElement):
    def __init__(self, input):
        self.input = input

    def _process_input(self, input):
        return
    
    def _execute(self):
        return 
    
    def get_result(self):
        return self.input
    
class MaskElement(AbstractPipelineElement):
    def __init__(self, mask: list[int | bool]):
        self.mask = mask
        return
    
    def _process_input(self, input):
        self.input = input
        return 
    
    def _execute(self):
        self.result = [elem for i, elem in enumerate(self.input) if self.mask[i]]

    def get_result(self):
        return self.result
    
class IterElement(AbstractPipelineElement):
    def __init__(self, elements: list[type[AbstractPipelineElement]], dbg=False, name=None):
        self.dbg = dbg
        self.name = name if name else self.__class__.__name__
        self.elements = elements
        return
    
    def _print(self, msg):
        print(f'{self.name}\t{msg}', flush=True)
    
    # Iterable -> In
    def _process_input(self, input):
        self.iterable_input = input
    
    def _execute(self):
        self.results = []
        self.pipes = []
        for i, obj in enumerate(self.iterable_input):
            p_name = self.name + f'-iter={i}'
            temp_elem = SequentialElement(self.elements, dbg=self.dbg, name=p_name)
            result = temp_elem._process(obj)
            self.results.append(result)
            self.pipes.append(temp_elem)
    
    # Out -> [elements(iter_elem) for iter_elem in Iterable]
    def get_result(self):
        return self.results
    
class TensorToDevice(AbstractPipelineElement):
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        return

    def _if_list(self, input):
        for i, elem in enumerate(input):
            input[i] = self._find_and_transfer(elem)

    def _if_dict(self, input):
        for key, value in input.items():
            input[key] = self._find_and_transfer(value)


    def _find_and_transfer(self, input):
        if isinstance(input, (list)):
            self._if_list(input)
        elif torch.is_tensor(input):
            return input.to(self.device)
        elif isinstance(input, (dict)):
            self._if_list(input.values())
        return input

    def _process_input(self, input):
        self.input = input

    def _execute(self):
        self._find_and_transfer(self.input)
    
    def get_result(self):
        return self.input
    


