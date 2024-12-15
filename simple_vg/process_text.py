from .commons import AbstractPipelineElement

# Text 데이터를 저장하는 Element
class SaveTextFile(AbstractPipelineElement):
    # 파일 쓰기만 할것이기 때문에 기본 open mode는 'w'임.
    def __init__(self, open_mode='w'):
        self.open_mode = open_mode
        return
    
    # 입력: [texts: [[텍스트묶음1....], [텍스트묶음2....]], out_paths: [묶음1의_경로, 묶음2의_경로]]
    def _process_input(self, input):
        self.texts = input[0]
        self.out_paths = input[1]

    # 단순히 묶음1 경로에 텍스트묶음1들의 텍스트를 라인별로 저장함.
    def _execute(self):
        for i, path in enumerate(self.out_paths):
            with open(path, self.open_mode) as f:
                for text in self.texts[i]:
                    f.writelines(text+'\n')
                f.flush()
                    
    # 출력: 없음
    def get_result(self):
        return None
    
# 흩어져있는 텍스트를 한줄로 정리해줌.
class ToTextDataset(AbstractPipelineElement):
    # key_order: 입력 dict에서 key_order에 명시된 key 순서대로 value를 순서대로 꺼내어 한줄의 텍스트로 만듦.
    # spliter: value들 사이에 들어갈 분리대임.
    # default: value가 없다면 대신 들어갈 텍스트임.

    def __init__(self, key_order: list, spliter='|', default=' '):
        self.key_order = key_order
        self.sp = spliter
        self.default = default

    # 입력: [dict]
    def _process_input(self, input):
        self.input = input
    
    # 한개의 dict당 한개의 str이 만들어짐.
    def _execute(self):
        lines = []
        for data_dict in self.input:
            data_dict: dict = data_dict
            line = self.sp.join([data_dict.get(key, self.default) for key in self.key_order])
            lines.append(line)
        self.result = lines
    
    # 출력: [str]
    def get_result(self):
        return self.result