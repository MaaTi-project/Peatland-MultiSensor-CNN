
class BestSelectionPlaceHolder:
    def __init__(self, id, channel, list_of_full_pathes, data):
        self.id = id
        self.channel = channel
        self.list_of_full_pathes = list_of_full_pathes
        self.list_of_relative_pathes = list_of_full_pathes
        self.data = data

    def stringify(self):
        return {
            'id': self.id,
            'channel': self.channel,
            'list_of_full_pathes': self.list_of_full_pathes,
            'list_of_relative_pathes': self.list_of_relative_pathes,
            'list_of_data': self.data.shape
        }


class InputSelectionPlaceHolder:
    def __init__(self, channel, id, full_path, satellite_name, image_name, label_set, data, shape, accuracy=-1, is_best=False):
        self.channel = channel
        self.id = id
        self.full_path = full_path
        self.relative_path = full_path
        self.satellite_name = satellite_name
        self.image_name = image_name
        self.data = data
        self.shape = shape
        self.label_set = label_set
        self.accuracy = accuracy 
        self.is_best = is_best

    def __eq__(self, other_object):
        if isinstance(other_object, InputSelectionPlaceHolder):
            return  self.channel == other_object.channel and self.full_path  == other_object.full_path
        return False

    def return_labels(self):
        return self.label_set

    def stringify(self):
        return {
            'id': self.id,
            'channel': self.channel,
            'full_path': self.full_path,
            'relative_path': self.relative_path,
            'data': self.data.shape,
            'label_set': len(self.label_set),
            'shape': self.shape,
            'accuracy': self.accuracy,
            'is best': self.is_best
        }
    
class AnalyseTheInputs: 
    def __init__(self, element_a: InputSelectionPlaceHolder, element_b: InputSelectionPlaceHolder):
        self.element_a = element_a
        self.element_b = element_b
    
    def similarity_check(self):
        if self.element_a.channel == self.element_b.channel and self.element_a.relative_path == self.element_b.relative_path:
            print(f"[SIMILARITY] the two inputs are the same")
            return True
        
    def check_best(self):
        if self.element_a.channel != self.element_b.channel:
            print(f"[SIMILARITY] Elements does not match channel wise")
            return False
        elif self.element_a.relative_path != self.element_b.relative_path:
            print(f"[SIMILARITY] Elements does not match path wise")
            return True 
        else:
            print(f"[SIMILARITY] Elements match")
            return False

    