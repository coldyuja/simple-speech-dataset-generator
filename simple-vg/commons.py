from typing import Generic
from vg_types import In, Out

NOT_IMPL_ERR_MSG = 'You MUST implement this method!!'

class AbstractPipelineElement(Generic[In, Out]):
    def __init__(self):
        return

    def _process_input(self, input: In):
        raise NotImplementedError(NOT_IMPL_ERR_MSG)

    def _execute(self):
        raise NotImplementedError(NOT_IMPL_ERR_MSG)

    def get_result(self):
        raise NotImplementedError(NOT_IMPL_ERR_MSG)
    
    def _process(self, input: In) -> Out:
        self._process_input(input)
        self._execute()
        return self.get_result()