NOT_IMPL_ERR_MSG = 'You MUST implement this method!!'

class AbstractPipelineElement:
    def __init__(self):
        return

    def _process_input(self, input):
        raise NotImplementedError(NOT_IMPL_ERR_MSG)

    def _execute(self):
        raise NotImplementedError(NOT_IMPL_ERR_MSG)

    def get_result(self):
        raise NotImplementedError(NOT_IMPL_ERR_MSG)
    
    def _process(self):
        self._execute()
        return self.get_result()