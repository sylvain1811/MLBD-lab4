

class ModelVar:

    def __init__(self, model_path, dict_freq_var):

        self.model_path=model_path
        self.dict_freq_var=dict_freq_var


    def __str__(self):
        string = "model path: {0} variables {1}".format(self.model_path, self.dict_freq_var)
        return string