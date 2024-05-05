# SED_fitter.py

import numpy as np
from astropy.table import Table
from abc import abstractmethod
from tqdm import tqdm
from joblib import Parallel, delayed
import astropy.units as u

from . import config, galfind_logger
from . import SED_code
from .decorators import run_in_dir

class SED_Fitter(SED_code):

    def __init__(self, galaxy_property_dict, galaxy_property_errs_dict, available_templates):
        super().__init__(galaxy_property_dict, galaxy_property_errs_dict, available_templates)

    @staticmethod
    def label_from_SED_fit_params(SED_fit_params):
        assert("code" in SED_fit_params.keys() and "templates" in SED_fit_params.keys())
        return f"{SED_fit_params['code'].__class__.__name__}_{SED_fit_params['templates']}"

    def SED_fit_params_from_label(self, label):
        templates = label.replace(f"{self.__class__.__name__}_", "")
        assert(templates in self.available_templates)
        return {"code": self, "templates": templates}

    def galaxy_property_labels(self, gal_property, SED_fit_params):
        assert("templates" in SED_fit_params.keys())
        assert(SED_fit_params["templates"] in self.available_templates)
        assert(gal_property in self.galaxy_property_dict.keys())
        return f"{self.galaxy_property_dict}_{SED_fit_params['templates']}"

    @abstractmethod
    def make_in(self, cat):
        pass
    
    @run_in_dir(path = config['SED_Fitter']['SED_FITTER_DIR'])
    def run_fit(self, in_path, fits_out_path, instrument, SED_fit_params, n_jobs = 6, overwrite = False):
        # sonora + [bobcat, cholla]
        # sonora_path='/nvme/scratch/work/tharvey/brown_dwarfs/'
        '''
        Fits general templates to a catalogue.
        
        Inputs: 
        n_jobs - int - number of parallel jobs to run. Default is 6.
        overwrite - bool - whether to rerun fitting if output columns found in catalog. 
            If rerun is False and columns are matched, fitting is not performed but 
            candidates will still be recomputed. Default is False.
        '''

        # read in templates
        sonora = Table.read(sonora_path+f'sonora_model/sonora_{model_version}.param', format='ascii', delimiter=' ', names=['num', 'path', 'scale'])
        
        # calculate mock photometry for each template

        models_table = Table()
        sonora_names = []
        # make SED template for each template in template path
        # create mock photometry for each of these templates

        # Read this file in and add if needed to the table - will be quicker than creating each time
        for pos, row in enumerate(sonora):
            name = row["path"].split("/")[-1]
            sonora_names.append(name)
            table = Table.read(sonora_path+row['path'], names=['wav', 'flux_nu'], format='ascii.ecsv', delimiter=' ', units=[u.AA, u.erg/(u.cm**2*u.s*u.Hz)])
            table['flux_njy'] = table['flux_nu'].to(u.nJy)/1e17
            convolved_fluxes = [i.value for i in convolve_sed_v2(table['flux_njy'], table['wav'].to(u.um), bands_all, input='flux')]
            #print(bands_all)
            #print(convolved_fluxes)
            flux_column = Column(convolved_fluxes, name=name, unit=u.nJy)
            models_table.add_column(flux_column)
        
        params = [(id, bands_id, catalog, id_col, models_table, sonora_names, band_wavs, bands_all, plot_all, plot, sonora_path, fwhm_band_size, size_compare_band, force_fit, absolute_chi2_good_fit) for id, bands_id in bands.items()]
        output = Parallel(n_jobs=n_jobs)(delayed(self.run_fit_single_core)(param) for param in tqdm(params, desc = f"Fitting {self.label_from_SED_fit_params(SED_fit_params)}"))
        output = np.array(output)
        
        #results_table = Table([list(bands.keys()), templates, chi2, constants], names=['id', 'template', 'chi2', 'constant'])
        
        print('Finished fitting. Saving results.')
        catalog.write(f'{catalog_path[:-5]}_sonora_{model_version}.fits', overwrite=True)
        print('Written output to ', f'{catalog_path[:-5]}_sonora_{model_version}.fits')
    
    @staticmethod # not sure how to pass class when parallelising, so keeping this method static for now and passing self in via params
    def run_fit_single_core(params):
        self, input_row, SED_templates, models_table, sonora_names, sonora_path = params
        # mask those columns with -99s in the input table
        flux = flux.to(u.nJy)
        flux_err = flux_err.to(u.nJy)
        consts = []
        fits = []
        for sonora_name in sonora_names: 
            models_mask = [pos for pos, band in enumerate(bands_all) if band in bands_id]
            popt, pcov = curve_fit(lambda x, a: a * models_table[sonora_name][models_mask], band_wavs, flux, sigma = flux_err, p0=1e-4)
            const = popt[0]
            chi_squared = np.sum(((const * models_table[sonora_name][models_mask] - flux) / flux_err)**2)
            consts.append(const)
            fits.append(chi_squared)
        
        best_fit = np.argmin(fits)
        
        #table = Table.read(f'{sonora_path}/sonora_model/{name}', names=['wav', 'flux_nu'], format='ascii.ecsv', delimiter=' ', units=[u.AA, u.erg/(u.cm**2*u.s*u.Hz)])
        #table['flux_njy'] = table['flux_nu'].to(u.nJy)/1e17
        #table['wav'], table['flux_njy'] * consts[best_fit]
        
        # name of best fit template, chi2 of best fit template and scaling constant of best fit template
        return sonora_names[best_fit], fits[best_fit], consts[best_fit]
    
    @abstractmethod
    def make_fits_from_out(self, out_path, SED_fit_params):
        pass
    
    @staticmethod
    @abstractmethod
    def get_out_paths(out_path, SED_fit_params, IDs):
        pass
    
    @staticmethod
    @abstractmethod
    def extract_SEDs(IDs, data_paths):
        pass
    
    @staticmethod
    @abstractmethod
    def extract_PDFs(gal_property, IDs, data_paths, SED_fit_params):
        pass

class Brown_Dwarf_Fitter(SED_Fitter):

    galaxy_property_dict = {key: key for key in ["chi2", "best_template"]}
    galaxy_property_errs_dict = {}
    available_templates = ["sonora_bobcat", "sonora_cholla"]

    def __init__(self):
        super().__init__(self.galaxy_property_dict, self.galaxy_property_errs_dict, self.available_templates)

    def make_in(self, cat):
        pass
    
    def make_fits_from_out(self, out_path, SED_fit_params):
        pass

    @staticmethod
    def get_out_paths(out_path, SED_fit_params, IDs):
        fits_out_path = ""
        PDF_paths = {} # no PDFs
        # open saved table and determine best
        fits_tab = Table.read(fits_out_path, character_as_bytes = False, memmap = True)
        # crop to just relevant IDs
        #fits_tab["best_template"]
        SED_paths = f"{config['SED_Fitter']['TEMPLATES_DIR']}/{SED_fit_params['templates']}/"
        return fits_out_path, PDF_paths, SED_paths
    
    @staticmethod
    def extract_SEDs(IDs, SED_paths):
        pass

    @staticmethod
    def extract_PDFs(gal_property, IDs, PDF_paths, SED_fit_params):
        pass