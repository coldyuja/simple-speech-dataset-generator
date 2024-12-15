# 해당 파일에서 사용할 모듈들을 Import함.
from typing import Generic
from .vg_types import In, Out

# 자식 class에서 일부 method 미구현시에 대한 오류 메세자
NOT_IMPL_ERR_MSG = 'You MUST implement this method!!'

# AbstractPipelineElement
# Pipeline에서 실행될 Element는 부모 클래스로 이 클래스를 가지게 된다.
# Pipeline은 _process_input(), _execute(), get_result()를 사용하여 Element를 실행시키는데
# 이 클래스는 해당 요소를 필수로 가져 이 클래스를 부모 클래스로 둔 클래스들은 반드시 해당 method를 구현하여야 한다.
# Element의 통일된 실행 절차를 위해 생성되었다.
# 또한 Abstract Class로써 실질적인 동작은 하지 않는다.
class AbstractPipelineElement(Generic[In, Out]):
    def __init__(self):
        return

    def _process_input(self, input: In):
        raise NotImplementedError(NOT_IMPL_ERR_MSG)

    def _execute(self):
        raise NotImplementedError(NOT_IMPL_ERR_MSG)

    def get_result(self) -> Out:
        raise NotImplementedError(NOT_IMPL_ERR_MSG)
    
    def _process(self, input: In) -> Out:
        self._process_input(input)
        self._execute()
        return self.get_result()
    
# 추가적으로 AbstractPipelineElement를 통해 구현한 Element에 대한 설명
# 이 패키지에서 구현된 Element들은 AudioSetting을 초기화 입력으로 받으며
# 각 Element만의 세부 설정은 AudioSetting['opt_settings'](이하 ElementSettings)에 있음.
# ElementSettings은 각 Element에서 필요한 설정들을 추가적으로 담고 있는다. 
# 그 중 Model 선택이나, Task 선택과 같이 명시할 수 있는 작업이 있는 경우 Enum을 사용하여
# 종류를 선택하도록 하는데, Enum을 사용할 경우 str을 입력을 받는것 보다 오타율이 줄고 
# 일부 개발도구(vscode)에서는 종류를 hint로써 보여줘 명칭을 전부 외울 필요가 없다는 장점이 있다.
# 또한 ElementSettings가 TypedDict로 구현되어 같은 장점을 가지기도 한다.
    


# ModelWrapper
# 다른 외부의 모델 또는 모델을 실행시키는 코드를 이 패키지 내에서 통일된 코드로 사용할 수 있도록,
# 사용하는 Wrapper 클래스이다. WhisperWrapper, SepReformerWrapper가 이에 해당한다.
# 동일하게 이 클래스로 wrap된 모델은 이 클래스를 부모 클래스로 가져 아래의 method로 해당 동작 수행이 가능하다.
# 미구현된 기능에 대해서는 Exception을 일으킨다. 또한 외부 요인에 따라 입력이 초기화 떄 진행될 수도 있고, 
# 따로 process_input()을 통해 입력을 받을 떄가 있다. 이는 Wrapper마다 방식이 다를것이다.
class ModelWrapper:
    def __init__(self):
        return
    
    def process_input(self, input):
        raise NotImplementedError(NOT_IMPL_ERR_MSG)
    
    def inference(self):
        raise NotImplementedError(NOT_IMPL_ERR_MSG)
    
    def train(self):
        raise NotImplementedError(NOT_IMPL_ERR_MSG)
    
    def get_result(self):
        raise NotImplementedError(NOT_IMPL_ERR_MSG)