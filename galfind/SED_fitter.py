# SED_fitter.py

from astropy.table import Table

from . import config, galfind_logger
from . import SED_code
from .decorators import run_in_dir

class SED_Fitter(SED_code):

    galaxy_property_dict = {key: key for key in ["chi2", "best_template"]}
    galaxy_property_errs_dict = {}
    available_templates = ["sonora_bobcat", "sonora_cholla"]

    def __init__(self):
        super().__init__(self.galaxy_property_dict, self.galaxy_property_errs_dict, self.available_templates)

    def galaxy_property_labels(self, gal_property, SED_fit_params):
        assert("templates" in SED_fit_params.keys())
        assert(SED_fit_params["templates"] in self.available_templates)
        assert(gal_property in self.galaxy_property_dict.keys())
        return f"{self.galaxy_property_dict}_{SED_fit_params['templates']}"

    def make_in(self, cat):
        pass

    @run_in_dir(path = config['SED_Fitter']['SED_FITTER_DIR'])
    def run_fit(self, in_path, fits_out_path, instrument, SED_fit_params, n_jobs = 6, overwrite = False):
         absolute_chi2_good_fit = 10):
        # sonora + [bobcat, cholla]
        # sonora_path='/nvme/scratch/work/tharvey/brown_dwarfs/'
        '''
        Fits sonora brown dwarf templates to a catalog.
        
        Inputs: 
        n_jobs - int - number of parallel jobs to run. Default is 6.
        overwrite - bool - whether to rerun fitting if output columns found in catalog. 
            If rerun is False and columns are matched, fitting is not performed but 
            candidates will still be recomputed. Default is False.

        Returns:

        catalog - astropy.table.Table instance - catalog with new columns added.
        Columns added are as follows:
            best_template_sonora_{model_version} - string - name of best fit template.
            chi2_best_sonora_{model_version} - float - chi2 of best fit template.
            constant_best_sonora_{model_version} - float - scaling constant of best fit template.
            delta_chi2_{compare_galaxy_chi2}_sonora_{model_version} - float - chi2 difference between best BD fit and galaxy fit if compare_galaxy_chi2 is not False.
            possible_brown_dwarf_chi2_sonora_{model_version} - bool - whether source is a possible BD based on good chi2.
            possible_brown_dwarf_chi2_sonora_{model_version}_compare - bool - whether source is a possible BD based on good chi2 and chi2 difference to galaxy fit.
            possible_brown_dwarf_compact_{model_version}_compare - bool - whether source is a possible BD based on good chi2, chi2 difference to galaxy fit and size.
        '''

    # read in templates
    sonora = Table.read(sonora_path+f'sonora_model/sonora_{model_version}.param', format='ascii', delimiter=' ', names=['num', 'path', 'scale'])
    
    # calculate mock photometry for each template

    models_table = Table()
    sonora_names = []
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
    output = Parallel(n_jobs=n_jobs)(delayed(parallel_sonora_fitting)(param) for param in tqdm(params, desc='Fitting BD templates'))
    output = np.array(output)
        

    #results_table = Table([list(bands.keys()), templates, chi2, constants], names=['id', 'template', 'chi2', 'constant'])
    
    print('Finished fitting. Saving results.')
    catalog.write(f'{catalog_path[:-5]}_sonora_{model_version}.fits', overwrite=True)
    print('Written output to ', f'{catalog_path[:-5]}_sonora_{model_version}.fits')

    @staticmethod
    def label_from_SED_fit_params(SED_fit_params):
        assert("code" in SED_fit_params.keys() and "templates" in SED_fit_params.keys())
        return f"{SED_fit_params['code'].__class__.__name__}_{SED_fit_params['templates']}"

    def SED_fit_params_from_label(self, label):
        templates = label.replace(f"{self.__class__.__name__}_", "")
        assert(templates in self.available_templates)
        return {"code": self, "templates": templates}
    
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