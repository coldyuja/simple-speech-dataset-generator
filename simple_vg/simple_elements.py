# 해당 파일에서 사용할 모듈들을 Import함.
from typing import Any, Callable, TypeVar
import torch
from simple_vg.pipeline import VGPipeline
from .commons import AbstractPipelineElement
from torch import Tensor
import numpy as np

# ClosureElement에서 parameter로 받는 함수의 signature을 정의했음.
# 나중에 사용자가 코드를 작성할 때 있는편이 도움되기 때문임.
input_t = TypeVar('Input')
output_t = TypeVar('Output')
input_fn_t = Callable[[type[AbstractPipelineElement], input_t], Any]
exec_fn_t = Callable[[type[AbstractPipelineElement], Any], output_t]

# Closure나 lambda 처럼 명시적으로 정의하지는 않지만 익명으로 간단한 작업을 할때 사용할 수 있음.
# 물론 Callable을 받으므로 일반 함수를 넘길수도 있음.
class ClosureElement(AbstractPipelineElement):
    def __init__(self, input_fn: input_fn_t, exec_fn: exec_fn_t, opt_data=None):
        self.input_fn = input_fn
        self.exec_fn = exec_fn
        # 나중에 필요할 경우 각 함수가 사용할 수 있음
        self.opt_data = opt_data

    # 각 input_fn, exec_fn은 ClosureElement와 input을 받게 되는데
    # self=ClosureElement를 함수 내에서 사용함으로써 class 내 저장도 가능하다.
    # 또한 opt_data가 class 내 저장될 수 있으므로 이것을 함수 내에서 사용도 가능하다.
    # input -> input_fn -> exec_fn -> output
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
    
# Sequential로 각 Element를 실행할 경우 일반적으로는 하나의 path만 존재하게 된다.
# 즉, Pipeline은 한개의 list[Element]를 받으므로 데이터의 진행 경로가 1개라는 것이다.
# 하지만 특정 Element의 출력을 여러개의 Element로 전달하고 싶을 떄가 있는데, 이때 ParallelElement를 사용한다. 
class ParallelElement(AbstractPipelineElement):
    # 이 Element는 list[list[Element]]를 받는데,  list내 각 list[Element]는 Sequential로 처리되고
    # ParallelElement의 입력을 그대로 전달 받는다.
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
    
    # 각 elem_list: list[Element]을 Pipeline으로 만들어 ParallelElement가 입력으로 받은 것을 그대로 전달받아 처리한다. 
    def _execute(self):
        # 각 elem_list의 결과는 차례대로 results에 들어가고 이는 ParallelElement의 최종 리턴값이다.
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
    
# 기본적인 동작은 Pipeline과 동일하고 그와 다른점은 AbstractPipelineElement라는 점이다.
# 일반적으로는 사용될 이유가 없다. 하지만 아래의 IterElement와 같은 경우에 내부적인 구현으로 사용된다. 
class SequentialElement(AbstractPipelineElement):
    # ParallelElement와 비슷하게 list[Element]를 입력으로 받고 그냥 Pipeline과 같이 이를 순차적으로 실행시키고 최종값을 리턴한다.
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
    

# 일종의 Pipeline에서 Padding 역할을 한다.
# 받은 입력을 그대로 출력으로 내보낸다.
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
    
# 입력을 받지 않고 초기화시 주어진 입력을 출력으로 내보낸다.
class InputElement(AbstractPipelineElement):
    def __init__(self, input):
        self.input = input

    def _process_input(self, input):
        return
    
    def _execute(self):
        return 
    
    def get_result(self):
        return self.input

# ParallelElement와 같이 여러개의 입력을 제공하는 경우(list[output])
# 그 출력 중 특정 인덱스의 원소만 입력으로 받고 싶을때가 있다.
# 이 때, MaskElement를 사용하면 특정 원소를 masking 하여 해당 원소만 제외한 채로 출력으로 내보낸다.
# Ex) [a, b, c] -> MaskElement([1, 0, 1]) -> [a, c]
class MaskElement(AbstractPipelineElement):
    def __init__(self, mask: list[int | bool]):
        self.mask = mask
        return
    
    def _process_input(self, input):
        self.input = input
        return 
    
    def _execute(self):
        # 여기서 mask가 1인 경우에만 원소를 다시 모은다.
        self.result = [elem for i, elem in enumerate(self.input) if self.mask[i]]

    def get_result(self):
        return self.result
    
# 어떤 Element는 list가 아닌 한개의 입력을 받을 때가 있다.
# 이 때, IterElement를 사용하면 주어진 Iterable한 입력을 분리하여 주어진 Element에 넣고 이를 다시 모아 출력한다.
# Ex) [a, b, c] -> IterElement([ClosureElement(lambda a:int : a+1)]) -> [a+1, b+1, c+1]
class IterElement(AbstractPipelineElement):
    def __init__(self, elements: list[type[AbstractPipelineElement]], dbg=False, name=None):
        self.dbg = dbg
        self.name = name if name else self.__class__.__name__
        self.elements = elements
        return
    
    def _print(self, msg):
        print(f'{self.name}\t{msg}', flush=True)
    
    # 입력: Iterable
    def _process_input(self, input):
        self.iterable_input = input
    
    def _execute(self):
        # 각 Iter 결과는 list로 모아 최종 리턴값으로 출력된다.
        self.results = []
        self.pipes = []
        # Iterable Object를 for문으로 처리한다
        for i, obj in enumerate(self.iterable_input):
            p_name = self.name + f'-iter={i}'
            temp_elem = SequentialElement(self.elements, dbg=self.dbg, name=p_name)
            result = temp_elem._process(obj)
            self.results.append(result)
            self.pipes.append(temp_elem)
    
    # 출력: [elements(iter_elem) for iter_elem in Iterable]
    def get_result(self):
        return self.results
    
# TensorToDevice는 torch 전용으로 입력 내 모든 Tensor를 명시된 torch.device로 보낸다.
class TensorToDevice(AbstractPipelineElement):
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        return

    # 재귀적으로 list와 dict의 값들을 탐색하고
    # Tensor가 발견되면 .to(device)하여 이를 리턴한다.
    def _find_and_transfer(self, input):
        if isinstance(input, (list)):
            self._if_list(input)
        elif torch.is_tensor(input):
            return input.to(self.device)
        elif isinstance(input, (dict)):
            self._if_list(input.values())
        return input

    # list인 경우 아래와 같이 탐색하고
    def _if_list(self, input):
        for i, elem in enumerate(input):
            input[i] = self._find_and_transfer(elem)

    # dict인 경우 아래와 같이 탐색한다.
    def _if_dict(self, input):
        for key, value in input.items():
            input[key] = self._find_and_transfer(value)

    def _process_input(self, input):
        self.input = input
        
    # 최종적으로는 탐색 가능한 모든 Tensor는 명시된 torch.device로 전송된다.
    def _execute(self):
        self.input = self._find_and_transfer(self.input)
    
    def get_result(self):
        return self.input
    


