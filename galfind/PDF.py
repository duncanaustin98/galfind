# PDF.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from . import config, galfind_logger
from . import useful_funcs_austind as funcs

class PDF:

    def __init__(self, property_name, x, p_x):
        self.property_name = property_name
        self.x = x
        # normalize to np.trapz(p_x, x) == 1
        self.p_x = p_x / np.trapz(p_x, x)

    def __str__(self):
        return f"LOADED PDF FOR {self.property_name}"

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
    
    def get_peak(self, nth_peak):
        # not properly implemented yet
        try:
            self.peaks[nth_peak]
        except (AttributeError, IndexError) as e:
            if type(e) == AttributeError:
                self.peaks = []
            # calculate the nth_peak - what if array isnt the correct length
            if nth_peak == 0:
                self.peaks.append({"value": None, "chi_sq": None})
        return self.peaks[nth_peak]
            
        # currently just copied straight from Tom's plotting script
        # # calculate peak locations etc - should go inside of PDF class
        # pz_column, integral, peak_z, peak_loc, peak_second_loc, secondary_peak, ratio = useful_funcs_updated_new_galfind.robust_pdf([gal_id], [zbest], SED_code, field_name, rel_limits=True, z_fact=int_limit, use_custom_lephare_seds=custom_lephare, template=template, plot=False, version=catalog_version, custom_sex=custom_sex, min_percentage_err=min_percentage_err, custom_path=eazy_pdf_path, use_galfind=True)
        # print(integral, 'integral', peak_z, 'peak_z', peak_loc, 'peak_loc', peak_second_loc, 'peak_second_loc', secondary_peak, 'secondary_peak', ratio, 'ratio')

    def get_percentile(self, percentile):
        assert type(percentile) in [float], \
            galfind_logger.critical(f"percentile = {percentile} with type(percentile) = {type(percentile)} is not in ['float']")
        try:
            return self.percentiles[f"{percentile:.1f}"]
        except (AttributeError, KeyError) as e:
            if type(e) == AttributeError:
                self.percentiles = {}
            # calculate percentile
            cdf = np.cumsum(self.p_x)
            cdf /= np.max(cdf)
            self.percentiles[f"{percentile:.1f}"] = float(self.x[np.argmin(np.abs(cdf - percentile / 100.))])
            return self.percentiles[f"{percentile:.1f}"]

    def plot(self, ax, annotate = True, annotate_peak_loc = False, colour = "black"):
        
        ax.plot(self.x, self.p_x, color = colour)
        
        # Set xlim to 1% and 99% of PDF cumulative distribution
        norm = np.cumsum(self.p_x)
        norm = norm / np.max(norm)
        lowz = self.x[np.argmin(np.abs(norm - 0.01))] #- 0.3
        highz = self.x[np.argmin(np.abs(norm - 0.99))] #+ 0.3
        #print(lowz, highz)
        #breakpoint()
        ax.set_xlim(lowz, highz)
        ax.set_ylim(0, 1.2 * np.max(self.p_x)) # * 1.2

        # fill inside PDF with hatch
        x_lim = np.linspace(lowz, highz) #np.linspace(0.93 * float(self.get_peak(0)["value"]), 1.07 * float(self.get_peak(0)["value"]), 100)
        pdf_lim = np.interp(x_lim, self.x, self.p_x)
        ax.fill_between(x_lim, pdf_lim, color = colour, alpha = 0.2, hatch = '//')

        ax.grid(False)

        if annotate:
            # Draw vertical line at zbest
            ax.axvline(self.get_peak(0)["value"], color = colour, linestyle='--', alpha=0.5, lw=2)
            ax.axvline(self.get_percentile(16.), color = colour, linestyle=':', alpha=0.5, lw=2)
            ax.axvline(self.get_percentile(84.), color = colour, linestyle=':', alpha=0.5, lw=2)
            ax.annotate('-1$\sigma$', (self.get_percentile(16.), 0.1), fontsize='small', ha='center', transform = ax.get_yaxis_transform(), va='bottom',  color = colour, path_effects = [pe.withStroke(linewidth=3, foreground='white')])
            ax.annotate('+1$\sigma$', (self.get_percentile(84.), 0.1), fontsize='small', ha='center', transform = ax.get_yaxis_transform(), va='bottom', color = colour, path_effects = [pe.withStroke(linewidth=3, foreground='white')])
            ax.annotate(r'$z_{\rm phot}=$'+f'{self.get_peak(0)["value"]:.1f}' + \
                f'$^{{+{(self.get_percentile(84.) - self.get_peak(0)["value"]):.1f}}}_{{-{(self.get_peak(0)["value"] - self.get_percentile(16.)):.1f}}}$', (self.get_peak(0)["value"], 1.17), fontsize='medium', va='top', ha='center', color = colour, path_effects = [pe.withStroke(linewidth=3, foreground='white')])
            
            # Horizontal arrow at PDF peak going left or right depending on which side PDF is on, labelled with chi2
            # Check if highest peak is closer to xlim[0] or xlim[1]
            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()
            amount = 0.3 * (x_lim[1] - x_lim[0])
            if self.get_peak(0)["value"] - x_lim[0] < x_lim[1] - self.get_peak(0)["value"]:
                direction = 1
            else:
                direction = -1
            ax.annotate(r'$\chi^2=$'+f'{self.get_peak(0)["chi_sq"]:.2f}', (self.get_peak(0)["value"], 1.0), \
                xytext = (self.get_peak(0)["value"] + direction * amount, 0.90),  fontsize='small', va='top', \
                ha='center', color = colour, path_effects=[pe.withStroke(linewidth=3, foreground='white')], \
                arrowprops = dict(facecolor = colour, edgecolor = colour, arrowstyle='-|>', lw=1.5, \
                path_effects=[pe.withStroke(linewidth=1, foreground='white')]))
            
            # annotate PDF with peak locations etc
            # if annotate_peak_loc:
            #     ax.scatter(self.get_peak(0)["value"], peak_pdf, color = colour, edgecolors = colour, marker='o', facecolor='none')
                
            #     secondary_peak = self.get_peak(1)["value"]
            #     if secondary_peak > 0: 
            #         ax.scatter(secondary_peak, secondary_peak_pdf, edgecolor='orange', marker='o', facecolor='none')
            #         ax.annotate(f'P(S)/P(P): {ratio:.2f}', loc_ratio, fontsize='x-small')
            
            #ax.annotate(f'$\\sum = {float(integral):.2f}$', (zbest, 0.45), fontsize='small', \
                #transform = ax.get_yaxis_transform(), va='bottom', ha='center', fontweight='bold', \
                #color=eazy_color, path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    

class SED_fit_PDF(PDF):

    def __init__(self, property_name, x, p_x, SED_fit_params):
        self.SED_fit_params = SED_fit_params
        super().__init__(property_name, x, p_x)

    def load_peaks_from_SED_result(self, SED_result, nth_peak = 0):
        assert type(nth_peak) == int, galfind_logger.critical(f"nth_peak with type = {type(nth_peak)} must be of type 'int'")
        assert nth_peak == 0, galfind_logger.critical(f"SED_fit_PDF.load_peaks_from_SED_result only loads the 0th peak, not the {funcs.ordinal(nth_peak)}")
        assert SED_result.SED_fit_params == self.SED_fit_params, \
            galfind_logger.critical(f"SED_result.SED_fit_params = {SED_result.SED_fit_params} != self.SED_fit_params = {self.SED_fit_params}")
        # load peak value and peak chi_sq
        self.load_peaks_from_best_fit(SED_result.properties[self.property_name], SED_result.properties["chi_sq"])
        return self

    def load_peaks_from_best_fit(self, property, chi_sq):
        zeroth_peak = {"value": property, "chi_sq": chi_sq}
        if not hasattr(self, "peaks"):
            self.peaks = []
        if len(self.peaks) > 0:
            self.peaks[0] = zeroth_peak
        else:
            self.peaks.append(zeroth_peak)
        return self

class Redshift_PDF(SED_fit_PDF):

    def __init__(self, z, p_z, SED_fit_params):
        super().__init__("z", z, p_z, SED_fit_params)

    @classmethod
    def from_SED_code_output(cls, data_path, ID, code):
        z, p_z = code.extract_z_PDF(data_path, ID)
        return cls(z, p_z)
    
    def integrate_between_lims(self, delta_z, zbest = None, z_min = float(config["SEDFitting"].get("Z_MIN")), z_max = float(config["SEDFitting"].get("Z_MAX"))):
        # find best fitting redshift from peak of the PDF distribution - not needed if peak is loaded in PDF object
        if type(zbest) == type(None):
            zbest = self.get_peak(0)["value"] # find first peak
        elif type(zbest) in [int, float]: # correct format
            pass
        else:
            galfind_logger.critical(f"zbest = {zbest} with type = {type(zbest)} is not in [int, float, None]!")
        # calculate redshift limits
        lower_z_lim = np.clip(zbest * (1 - delta_z), z_min, z_max)
        upper_z_lim = np.clip(zbest * (1 + delta_z), z_min, z_max)
        return super().integrate_between_lims(lower_z_lim, upper_z_lim)


