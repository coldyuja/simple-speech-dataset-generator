from typing import Any, Callable, TypeVar
from commons import AbstractPipelineElement

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