# Multiple_Catalogue.py

class Multiple_Catalogue:

    def __init__(self, cat_arr, survey):
        self.cat_arr = cat_arr
        self.survey = survey

    @classmethod
    def from_pipeline(cls):
        cat_arr = []
        return cls(cat_arr)

    def calc_UVLF(self):
        pass

    def calc_GSMF(self):
        pass

    def plot(self, x_name, y_name, colour_by, save = False, show = False):
        pass