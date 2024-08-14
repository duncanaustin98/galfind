import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import NoReturn, Union
import astropy.units as u
import os

from .SED import SED_obs
from .SED_codes import SED_code
from . import Galaxy, Photometry_obs
from . import useful_funcs_austind as funcs
from . import galfind_logger, config, astropy_cosmo

class Number_Density_Function:

    def __init__(self, x_name, x, x_bin_edges, phi, phi_errs):
        self.x_name = x_name
        self.x = x
        self.x_bin_edges = x_bin_edges
        self.phi = phi
        self.phi_errs = phi_errs

    @classmethod
    def from_cat(cls, cat, x_name: str, x_origin: Union[str, dict], z_bin: Union[list, np.array], \
            SED_fit_params: Union[dict, str] = "EAZY_fsps_larson_zfree", z_step: float = 0.1, \
            use_vmax_simple: bool = False, timed: bool = False) -> "Number_Density_Function":
    #         def mass_function(catalog, fields, z_bins, mass_bins, rerun=False, out_directory = '/nvme/scratch/work/tharvey/masses/',
    #  mass_keyword='MASS_BEST',mass_form='log', z_keyword='Z_BEST', sed_tool='LePhare', template='', z_step=0.01,
    #   n_jobs=2, cat_version='v7', do_muv=False, use_vmax_simple = False, field_keyword='field', 
    #   other_name = '', other_sed_path='/nvme/scratch/work/austind/Bagpipes/pipes/seds/', 
    #   use_base=True, base_cat='/nvme/scratch/work/tharvey/catalogs/robust_and_good_gal_all_criteria_3sigma_all_fields_masses.fits',
    #   id_keyword='NUMBER',  use_new_zloop=True, select_444=False, use_bootstrap=True, rerun_other_pdfs = True,
    #     other_appended=False, flag_ids=[], base_cat_filter=None, zgauss=False):
    
        # calculate Vmax for each galaxy in catalogue within z bin
        # in general call Vmax_multifield
        cat.calc_Vmax(cat.data, z_bin, SED_fit_params, z_step, timed = timed)
        # extract x_name values from catalogue
        x = getattr(cat, x_name, x_origin)
        
        if use_vmax_simple:
            vmax_keyword = 'V_max_simple'
        else:
            vmax_keyword = 'V_max'

        if do_muv:
            muv = '_MUV'
        else:
            muv = ''

        # if use_bootstrap:
        #     btstrp_keyword = 'btstrp'
        # else:
        #     btstrp_keyword = ''

        # if not do_muv:
        #     # mass function?
        #     # extract masses
        #     masses = None # these should be linear
        #     # do mass correction
        #     ext_src_corr = None
        #     masses *= ext_src_corr       
        # else:
        #     mass = catalog[mass_keyword].data
        # "mass" could also be mUV in this circumstance
            
        if type(x_origin) in [dict]:
            assert "code" in x_origin.keys()
            assert x_origin["code"].__class__.__name__ in [code.__name__ for code in SED_code.__subclasses__()]
        
        # calculate optimal redshift bin size?
        # try:
        #     zs = np.array([i[0] for i in catalog[z_keyword]])
        #     for z_bin in z_bins:
                
        #         mask = (zs > z_bin[0]) & (zs < z_bin[1])
        #         mass_test_z = mass_test[mask]
        #         iqr = np.subtract(*np.percentile(mass_test_z, [75, 25]))

        #         print(f'Optimal bin size: {2*iqr/np.cbrt(len(mass_test_z)):.2f}')
        # except:
        #     pass
        
        # construct redshift lists and mass bin ragged nested lists 
        #bin_array = np.zeros(shape=(len(z_bins), len(mass_bins)))
        bin_array = [[[] for j in range(len(mass_bins[tuple(i)]))] for pos, i in enumerate(z_bins)]
        z_bin_mid = [(i+j)/2 for i,j in z_bins]
        z_bin_start = [i for i, j in z_bins]
        z_bin_end = [j for i, j in z_bins]
        len_array = [[[] for j in range(len(mass_bins[tuple(i)]))] for pos, i in enumerate(z_bins)]
        # fill arrays with redshifts and masses
        z = np.array([i for i in catalog[z_keyword]])
        for i, z_bin in enumerate(z_bins):
            for j, mass_bin in enumerate(mass_bins[tuple(z_bin)]):
                
                mass = np.array(mass)
                
                print(mass, mass_bin[0])
                bin_ids = list(catalog[id_keyword][(mass > mass_bin[0]) & (mass < mass_bin[1]) &  (z > z_bin[0]) & (z < z_bin[1])].value)

                bin_array[i][j] = bin_ids
                len_array[i][j] = len(bin_ids)
        print(bin_array)

        # plot length of arrays in each bin (separate!)
        # fig, ax_list = plt.subplots(nrows=len(bin_array), ncols=1, sharex=True)
        # if type(ax_list) !=np.ndarray:
        #     ax_list = [ax_list]
        # for pos, ax in enumerate(ax_list):
        #     z = z_bins[pos]
        #     bin_widths = [j-i for i,j in mass_bins[tuple(z)]]
        #     bin_start = [i for i, j in mass_bins[tuple(z)]]
        #     ax.bar(x=bin_start, height=len_array[pos], width=bin_widths, align='edge')
        #     ax.set_ylim(0, np.max(len_array[pos]))
        #     ax.annotate(f'z ~ {z_bin_mid[pos]}', (bin_start[0], 0.7 * ax.get_ylim()[1]))
        # if not do_muv:
        #     ax.set_xscale('log')
        #     ax_list[-1].set_xlabel('Galaxy Stellar Mass $(\log_{10} \ M_{\star}/M_{\odot})$')
        # else:
        #     ax_list[-1].set_xlabel('UV Absolute Magnitude (ABmag)')
        #fig.savefig(f'{out_directory}/z_mass_hist_{sed_tool}{muv}{name_444}.png')
        
        #logfile = f'{out_directory}/log_{sed_tool}{muv}{name_444}.txt'

        # sort completeness separately
        #catalog['jag_completeness'] = 0.0000
        #catalog['jag_contamination'] = 0.0000
        #catalog['total_completeness'] = 0.0000
        #catalog['detect_completeness'] = 0.0000

        # calculate mass function in each redshift bin
        for pos, redshift_bin in tqdm(enumerate(bin_array)):
            z_bin_min = z_bin_start[pos]
            z_bin_max = z_bin_end[pos]
            print(f'Doing {redshift_bin}')
            # loop through each mass bin in the given redshift bin
            for pos2, mass_ids in enumerate(redshift_bin):
            
                #print(mass_ids)
                if mass_ids != []:
                    parameters_list = []
                    #completeness_loop =[]
                    #jag_completeness_loop = []
                    #jag_contamination_loop = []
                
                    # loop through galaxies in z, mstar bin
                    # setup Vmax calculation

                    v_max = [float(i.value) for i,j,k,l in output] * u.Mpc**3
                    v_max_new = [float(j.value) for i,j,k,l in output] * u.Mpc**3

                    # determine number of fields used (should be easier in galfind)
                    num_fields_used = [k for i,j,k,l in output]
                    fields_used = [l for i,j,k,l in output]
                    
                    #filt = catalog['NUMBER'] == np.array(mass_ids)
                    filt = [True if id in mass_ids else False for id in catalog[id_keyword]]
                    
                    d_params, j_params, j_contam_params = [], [], []
                    for pos, id in enumerate(mass_ids):
                        d_params.append([fields_used[pos], completeness_loop[pos]])
                        j_params.append([fields_used[pos], jag_completeness_loop[pos]])
                        j_contam_params.append([fields_used[pos], jag_contamination_loop[pos]])

                        #d_comp = calculate_volume_weighted_comp(fields_used[pos], completeness_loop[pos])
                        #j_comp = calculate_volume_weighted_comp(fields_used[pos], jag_completeness_loop[pos])
                        #j_contam = calculate_volume_weighted_comp(fields_used[pos], jag_contamination_loop[pos], is_contam=True)
                        
                    average_j_comp = Parallel(n_jobs=n_jobs)(delayed(calculate_volume_weighted_comp)(params[0], params[1]) for params in j_params)    
                    average_j_contam = Parallel(n_jobs=n_jobs)(delayed(calculate_volume_weighted_comp)(params[0], params[1], is_contam=True) for params in j_contam_params)
                    average_d_comp = Parallel(n_jobs=n_jobs)(delayed(calculate_volume_weighted_comp)(params[0], params[1]) for params in d_params)
                    

                    catalog['V_max'][filt] = v_max
                    catalog['V_max_simple'][filt] = v_max_new
                    catalog['Num_fields'][filt] = num_fields_used
                    catalog['fields'][filt] = fields_used
                    catalog['detect_completeness'][filt] = average_d_comp
                    
                    mass_bin = mass_bins[(z_bin_min, z_bin_max)][pos2]
                    #catalog['num_fields_used'][filt] = fields_used
                    if not do_muv:
                        print(f'{np.log10(mass_bin[0])}_{np.log10(mass_bin[1])}')

                        catalog['mass_bin'][filt] = f'{np.log10(mass_bin[0])}_{np.log10(mass_bin[1])}'
                    else:
                        catalog['mass_bin'][filt] = f'{mass_bin[0]}_{mass_bin[1]}'

                    catalog['z_bin'][filt] = f'{z_bin_min}_{z_bin_max}'

                    #catalog['detect_completeness'][filt] = '_'.join(completeness_loop)
                    catalog['jag_completeness'][filt] = average_j_comp #'_'.join(jag_completeness_loop)
                    temp = np.array(average_j_comp) * np.array(average_d_comp)
                    temp[temp <= 0] = np.array(average_d_comp)[temp <= 0]

                    catalog['total_completeness'][filt] = temp
                    catalog['jag_contamination'][filt] = average_j_contam 
                    #mask = catalog['total_completeness'] < 0
                    #print(len(filt), len(mask))
                    # Replace total completeness with detection completeness if jag completeness is negative
                    #catalog['total_completeness'][mask & filt] = np.array(completeness_val)[catalog['total_completeness'][filt] < 0]
        catalog.write(f'/nvme/scratch/work/tharvey/masses/robust_good_all_gal_vmax_{sed_tool}{muv}{comp_contam}.fits', overwrite=True)

    print(f'/nvme/scratch/work/tharvey/masses/robust_good_all_gal_vmax_{sed_tool}{muv}{comp_contam}.fits')
    catalog = Table.read(f'/nvme/scratch/work/tharvey/masses/robust_good_all_gal_vmax_{sed_tool}{muv}{comp_contam}.fits', character_as_bytes=False)

    catalog = catalog[(catalog['V_max'] > 0) & (catalog['total_completeness'] > comp_limit)]

    if use_bootstrap:
    
        bootstrap_bins(catalog, fields, z_bins,mass_bins, len_array,
        rerun_other_pdfs, out_directory, z_keyword=z_keyword,
        mass_keyword=mass_keyword, mass_form=mass_form,
        field_keyword=field_keyword, other_name=other_name,
        other_sed_path=other_sed_path, load_duncans=load_duncans,
        id_keyword=id_keyword, other_h5_path=other_h5_path, muv=muv, 
        vmax_keyword=vmax_keyword, name_444=name_444, zgauss=zgauss, sed_tool=sed_tool)  

        plot_bins_pdfs(catalog, fields, z_bins, mass_bins, rerun=False, out_directory = out_directory,
        mass_form=mass_form,mass_keyword=mass_keyword, z_keyword=z_keyword, zgauss=zgauss)

    else:
        for pos, redshift_bin in enumerate(bin_array):
            fig, ax = plt.subplots(1,1)
            phi_list = []
            mass_list = []
            phi_error_list = []
            phi_error_cv_list = []
            num_bin = []
            avg_completeness = []
            avg_contamination = []
            mass_lower = []
            mass_higher = []
            cv_error_list = []    
            red_bin = redshift_bin
            for pos2, mass_ids in enumerate(red_bin):

                '''
                This doesn't work because far too many galaxies in volume.
                Need to draw lots of samples of same size as in each redshift bin and do mass function for each,
                and then plot range.
                From Cosmos2020 paper -  individual mass likelihood distributions to draw 1 000
                independent realizations of the galaxy stellar mass function and thereby directly estimate 
                the variance produced by the mass uncertainties, which we take as the 68% range about the
                median number density per bin of mass.
                Unclear if I actually need to bootstrap or just draw one mass for each galaxy lots of times. 

                if use_bootstrap:
                    bin_catalog = []
                    for var_id in mass_ids:  
                        bin_catalog.append(catalog[catalog[id_keyword] == var_id])
                    print('here')
                    print(len(bin_catalog))
                    bin_catalog = Table(bin_catalog)
                    print(len(bin_catalog))
                else:
                '''
                filt = [True if id in mass_ids else False for id in catalog[id_keyword]]
                print(len(catalog))
                bin_catalog = catalog[filt]
                print(len(bin_catalog))

                if len(bin_catalog) > 0:
                    
                    V_max = bin_catalog[vmax_keyword] * u.Mpc**3
                    completeness = bin_catalog['total_completeness']
                    #print(mass_bin)
                    #print(completeness)
                    
                    # Need to get mass bin from code 
                    #mass_bin = bin_catalog['mass_bin'][0].split('_')
                    z_bin = bin_catalog['z_bin'][0].split('_')
                    
                    print(mass_bin)
                    if not do_muv:
                        mass_bin = np.log10(mass_bins[(float(z_bin[0]), float(z_bin[1]))][pos2])
                    else:
                        mass_bin = mass_bins[(float(z_bin[0]), float(z_bin[1]))][pos2]
                
                    mass_bin_high = float(mass_bin[1])
                    mass_bin_low = float(mass_bin[0])
                    fields_used =  list(set(i for j in [i for i in [i.split('_') for i in bin_catalog['fields']]] for i in j))
                
                    mass_width = (mass_bin_high - mass_bin_low)
                    mass_bin_center = (mass_bin_high + mass_bin_low)/2
                    
                    print(mass_bin_low, mass_bin_center, mass_bin_high, mass_width)
                    V_max_err = np.sum(1/(completeness*V_max)**2)
                    phi = 1/mass_width * np.sum(1/(completeness*V_max))
                    phi_error = np.sqrt(V_max_err)/mass_width 
                    if len(V_max) < 4:
                        length = len(V_max)
                        phi_error = phi * np.min(np.abs((np.array(useful_funcs.poisson_interval(length, 0.32))-length))/length)
                    
                    cv = calc_cv_proper(float(z_bin[0]), float(z_bin[1]), fields_used=fields_used)
                    cv_error_list.append(cv)
                    phi_error_cv = np.sqrt(phi_error**2 + (cv*phi)**2)
                    phi_error_cv_list.append(phi_error_cv.value)
                    mass_lower.append(mass_bin_low)
                    mass_higher.append(mass_bin_high)
                    phi_list.append(phi.value)
                    phi_error_list.append(phi_error.value)
                    mass_list.append(mass_bin_center)
                    num_bin.append(len(V_max))
                    avg_completeness.append(np.mean(completeness))
                    avg_contamination.append(np.mean(bin_catalog['jag_contamination']))
                    ax.errorbar(mass_bin_center, phi, yerr = phi_error_cv, linestyle='none',marker='o', color='blue')
            out = np.vstack([mass_list, phi_list, phi_error_list, phi_error_cv_list, mass_lower, mass_higher, num_bin, avg_completeness, avg_contamination, cv_error_list]).T
            
            if not do_muv:
                ax.set_xlabel('Galaxy Stellar Mass $(\log_{10} \ M_{\star}/M_{\odot})$')
            else:
                ax.set_xlabel('UV Absolute Magnitude (ABmag)')

            ax.set_ylabel(f'$\Phi$ (dex$^{{-1}}$ Mpc$^{{-3}}$)')
            ax.set_yscale('log')
            #print(z_bin)
            redshift_bin = (float(z_bin[0])+float(z_bin[1]))/2
            ax.set_title(f'z~{redshift_bin}')
            fig.savefig(f'/nvme/scratch/work/tharvey/masses/debug/z_{redshift_bin}_{sed_tool}{muv}{name_444}{comp_contam}.png', dpi=100)
            
            np.savetxt(f'/nvme/scratch/work/tharvey/masses/out_mass_function_{vmax_keyword}_{sed_tool}_{redshift_bin}{muv}{name_444}{btstrp_keyword}{comp_contam}.dat', out, header='Log_Mass Phi Phi_Err Phi_Err_CV Mass_Bin_Low Mass_Bin_High NumBin AvgCompleteness AvgContamination CV_err', fmt='%.6e',)