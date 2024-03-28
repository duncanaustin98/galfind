# Multiple_Data.py

import numpy as np

class Multiple_Data:

    def __init__(self, data_arr):
        self.data_arr = data_arr

    @classmethod
    def from_pipeline(cls, surveys, versions, instrument_names_arr, excl_bands_arr):
        if type(versions) == str:
            versions = [versions for i in range(len(surveys))]
        assert(len(versions) == len(surveys))

        if type(instrument_names_arr) == str:
            instrument_names_arr = [[instrument_names_arr] for i in range(len(surveys))]
        elif type(instrument_names_arr) in [list, np.array]:
            if type(instrument_names_arr[0]) == str:
                instrument_names_arr = [instrument_names_arr for i in range(len(surveys))]
            elif type(instrument_names_arr[0]) in [list, np.array]:
                # already formatted correctly
                pass
            else:
                raise(Exception())
        else:
            raise(Exception())
        assert(len(instrument_names_arr) == len(surveys))

        if type(excl_bands_arr) == str:
            excl_bands_arr = [[excl_bands_arr] for i in range(len(surveys))]
        elif type(excl_bands_arr) in [list, np.array]:
            if type(excl_bands_arr[0]) == str:
                excl_bands_arr = [excl_bands_arr for i in range(len(surveys))]
            elif type(excl_bands_arr[0]) in [list, np.array]:
                # already formatted correctly
                pass
            else:
                raise(Exception())
        else:
            raise(Exception())
        assert(len(excl_bands_arr) == len(surveys))

        print(versions, excl_bands_arr)
        data_arr = []
        return cls(data_arr)
    
    @staticmethod
    def check_available_data(versions):
        # print available data for each version


if __name__ == "__main__":
    Multiple_Data.from_pipeline([["1"], ["2"]], "v9", "NIRCam", [["f090W", "f115W"], "f115W", "f115W"])
