from .commons import AbstractPipelineElement


class SaveTextFile(AbstractPipelineElement):
    def __init__(self, open_mode='w'):
        self.open_mode = open_mode
        return
    
    # [texts: [[bunch_of_txt1....], [txt2....]], out_paths: [path_of_txt1, path_of_txt2]] -> In
    def _process_input(self, input):
        self.texts = input[0]
        self.out_paths = input[1]

    def _execute(self):
        for i, path in enumerate(self.out_paths):
            with open(path, self.open_mode) as f:
                for text in self.texts[i]:
                    f.writelines(text+'\n')
                f.flush()
                    
        
    def get_result(self):
        return None
    

class ToTextDataset(AbstractPipelineElement):
    def __init__(self, key_order: list, spliter='|', default=' '):
        self.key_order = key_order
        self.sp = spliter
        self.default = default

    # input: [dict] -> In
    def _process_input(self, input):
        self.input = input
    
    def _execute(self):
        lines = []
        for data_dict in self.input:
            data_dict: dict = data_dict
            line = self.sp.join([data_dict.get(key, self.default) for key in self.key_order])
            lines.append(line)
        self.result = lines
    
    # Out -> [str]
    def get_result(self):
        return self.result