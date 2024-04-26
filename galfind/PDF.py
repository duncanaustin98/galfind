# PDF.py

import numpy as np
import matplotlib.pyplot as plt

from . import config, galfind_logger

class PDF:

    def __init__(self, property_name, x, p_x):
        self.property_name = property_name
        self.x = x
        # normalize to np.trapz(p_x, x) == 1
        self.p_x = p_x / np.trapz(p_x, x)

    @classmethod
    def from_1D_arr(cls):
        return NotImplementedError

    def integrate_between_lims(self, lower_x_lim, upper_x_lim):
        # find index of closest values in self.x to lower_x_lim and upper_x_lim
        index_x_min = np.argmin(np.absolute(self.x - lower_x_lim))
        index_x_max = np.argmin(np.absolute(self.x - upper_x_lim))
        # clip x/p_x distribution to integration limits
        x = self.x[index_x_min : index_x_max]
        p_x = self.p_x[index_x_min : index_x_max]
        # integrate using trapezium rule between limits
        return np.trapz(p_x, x)
    
    def find_peak(self, nth_peak):
        return NotImplementedError

    def plot_PDF(self, fig, ax):
        return NotImplementedError
    
class Redshift_PDF(PDF):

    def __init__(self, z, p_z):
        super().__init__("z", z, p_z)

    @classmethod
    def from_SED_code_output(cls, data_path, ID, code):
        z, p_z = code.extract_z_PDF(data_path, ID)
        return cls(z, p_z)
    
    def integrate_between_lims(self, delta_z, zbest = None, z_min = config["SEDFitting"].get("Z_MIN"), \
            z_max = config["SEDFitting"].get("Z_MAX")):
        # find best fitting redshift from peak of the PDF distribution
        if type(zbest) == type(None):
            zbest = self.find_peak(0) # find first peak
        elif type(zbest) in [int, float]: # correct format
            pass
        else:
            galfind_logger.critical(f"zbest = {zbest} with type = {type(zbest)} is not in [int, float, None]!")
        # calculate redshift limits
        lower_z_lim = np.clip(zbest * (1 - delta_z), z_min, z_max)
        upper_z_lim = np.clip(zbest * (1 + delta_z), z_min, z_max)
        return super().integrate_between_lims(lower_z_lim, upper_z_lim)


