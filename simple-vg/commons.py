NOT_IMPL_ERR_MSG = 'You MUST implement this method!!'

class AbstractPipelineElement:
    def __init__(self):
        return

    def _execute(self):
        raise NotImplementedError(NOT_IMPL_ERR_MSG)

    def get_result(self):
        raise NotImplementedError(NOT_IMPL_ERR_MSG)