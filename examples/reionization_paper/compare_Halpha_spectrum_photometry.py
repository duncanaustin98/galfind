
import galfind
from galfind import useful_funcs_austind as funcs
from galfind import astropy_cosmo as cosmo

from lmfit import Model, Parameters, minimize, fit_report
#from lmfit import create_params, fit_report, minimize
from astropy.table import Table, join
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

from ppxf.ppxf import ppxf
import ppxf.sps_util as lib
import ppxf.ppxf_util as util
import astropy.units as u
import astropy.constants as const
from urllib import request
from pathlib import Path

global failed_fits
failed_fits = []

def write_high_res_Halpha_tab():
    spec_cat = galfind.Spectral_Catalogue.from_DJA(
        version = "v3", z_cat_range = [5.5, 6.5], grade = 3
    )
    #medium_res = [spec.meta["FILENAME"] for spec_arr in spec_cat for spec in spec_arr if spec.instrument.grating.name == "G395M"]
    high_res = [spec.origin.split("/")[-1] for spec_arr in spec_cat for spec in spec_arr if spec.instrument.grating.name == "G395M"]
    grating_names = ["G395M" for spec in high_res]
    #grating_names.extend(["G395H" for spec in high_res])
    filenames = high_res
    #filenames.extend(high_res)
    tab = Table({"grating": grating_names, "filename": filenames})
    tab.write("G395M,56z65.ecsv", format = "ascii", overwrite  = True)

def make_nircam_filterset(survey):
    if survey == "JOF":
        filters = ['F090W', 'F115W', 'F150W', 'F162M', 'F182M', 'F200W', 'F210M', 'F250M', 'F277W', 'F300M', 'F335M', 'F356W', 'F410M', 'F444W']
        for i, filt_ in enumerate(filters):
            filt = galfind.Filter.from_filt_name("JWST/NIRCam." + filt_)
            if i == 0:
                filterset = filt
            else:
                filterset += filt
    return filterset

def get_depths(survey, version, filterset, depth_region = "all"):
    depth_path = f"{galfind.config['Depths']['DEPTH_DIR']}/Depth_tables/{version}/{survey}/{survey}_depths.ecsv"
    depth_tab = Table.read(depth_path)
    depths = [float(depth_tab[(depth_tab["filter"] == filt.band_name) & (depth_tab["region"] == depth_region)]["median_depth"]) for filt in filterset]
    return depths

def get_prism_cat_path(z_label, survey, version):
    return f"PRISM,{z_label}_{survey}_{version}.fits"

def get_zlabel(z_range):
    return f"{str(z_range[0]).replace('.', '_')}<z<{str(z_range[1]).replace('.', '_')}"

def make_prism_cat(survey = "JOF", version = "v11",depth_region = "all", z_range = [5.6, 6.5]):
    z_label = get_zlabel(z_range)
    filterset = make_nircam_filterset(survey)
    #filterset = galfind.Filter.from_filt_name("JWST/NIRCam.F410M") + galfind.Filter.from_filt_name("JWST/NIRCam.F444W")
    depths = get_depths(survey, version, filterset, depth_region = depth_region)
    
    spec_cat = galfind.Spectral_Catalogue.from_DJA(
        version = "v3", z_cat_range = z_range, grade = 3
    )
    prism_cat = [spec for spec_arr in spec_cat for spec in spec_arr if spec.instrument.grating.name == "PRISM"]
    prism_seds = [galfind.SED_obs(spec.z, spec.wavs.value, spec.fluxes.value, spec.wavs.unit, spec.fluxes.unit) for spec in prism_cat]
    mock_phot_arr = [sed.create_mock_phot(filterset, depths = depths) for sed in \
        tqdm(prism_seds, total = len(prism_seds), desc = "Making mock photometry")]
    scattered_mock_phot_arr = [mock_phot.scatter(1) for mock_phot in mock_phot_arr]
    phot_dict = {f"{filt.band_name}_Jy": [mock_phot.flux[i].value for mock_phot in mock_phot_arr] for i, filt in enumerate(filterset)}
    scattered_phot_dict = {f"{filt.band_name}_Jy_scattered": [mock_phot.flux[i].value for mock_phot in scattered_mock_phot_arr] for i, filt in enumerate(filterset)}
    phot_errs_dict = {f"{filt.band_name}_Jy_errs": [mock_phot.flux_errs[i].value for mock_phot in mock_phot_arr] for i, filt in enumerate(filterset)}
    ID = list(np.array(range(1, len(prism_cat) + 1)).astype(str))
    z = [spec.z for spec in prism_cat]
    filenames = [spec.origin.split("/")[-1] for spec in prism_cat]
    filepaths = [spec.origin for spec in prism_cat]
    tab = Table({"ID": ID, "z": z, "filenames": filenames, "filepaths": filepaths, **phot_dict, **scattered_phot_dict, **phot_errs_dict})
    tab.write(get_prism_cat_path(z_label, survey, version), format = "ascii", overwrite  = True)

def plot_prism_data(z_range):
    spec_cat = galfind.Spectral_Catalogue.from_DJA(
        version = "v3", z_cat_range = z_range, grade = 3
    )
    [spec.plot(out_dir = f"{get_zlabel(z_range)}_plots/") for spec_arr in spec_cat for spec in spec_arr if spec.instrument.grating.name == "PRISM"]


def fit_cat_Halpha(survey, version, z_range, fit_type):
    zlabel = get_zlabel(z_range)
    prism_cat_path = get_prism_cat_path(zlabel, survey, version)
    cat = Table.read(prism_cat_path, format = "ascii")
    spec_filepaths = list(cat["filepaths"])
    z_arr = list(cat["z"])
    Halpha_fit_arr = [fit_Halpha(z, spec_filepath, fit_type) for z, spec_filepath in zip(z_arr, spec_filepaths)]
    out_dict = {
        "spec_filepaths": spec_filepaths,
        "Ha_EWrest": [Halpha_fit.EWrest["EWrest_50"] if Halpha_fit is not None else np.nan for Halpha_fit in Halpha_fit_arr], \
        "Ha_EWrest_l1": [Halpha_fit.EWrest["EWrest_50"] - Halpha_fit.EWrest["EWrest_16"] if Halpha_fit is not None else np.nan for Halpha_fit in Halpha_fit_arr], \
        "Ha_EWrest_u1": [Halpha_fit.EWrest["EWrest_84"] - Halpha_fit.EWrest["EWrest_50"] if Halpha_fit is not None else np.nan for Halpha_fit in Halpha_fit_arr], \
        "Ha_cont": [Halpha_fit.cont["cont_50"] if Halpha_fit is not None else np.nan for Halpha_fit in Halpha_fit_arr],
        "Ha_cont_l1": [Halpha_fit.cont["cont_50"] - Halpha_fit.cont["cont_16"] if Halpha_fit is not None else np.nan for Halpha_fit in Halpha_fit_arr],
        "Ha_cont_u1": [Halpha_fit.cont["cont_84"] - Halpha_fit.cont["cont_50"] if Halpha_fit is not None else np.nan for Halpha_fit in Halpha_fit_arr],
        "Ha_flux": [Halpha_fit.Halpha_flux["Halpha_flux_50"] if Halpha_fit is not None else np.nan for Halpha_fit in Halpha_fit_arr],
        "Ha_flux_l1": [Halpha_fit.Halpha_flux["Halpha_flux_50"] - Halpha_fit.Halpha_flux["Halpha_flux_16"] if Halpha_fit is not None else np.nan for Halpha_fit in Halpha_fit_arr],
        "Ha_flux_u1": [Halpha_fit.Halpha_flux["Halpha_flux_84"] - Halpha_fit.Halpha_flux["Halpha_flux_50"] if Halpha_fit is not None else np.nan for Halpha_fit in Halpha_fit_arr],
        "Ha_SNR": [Halpha_fit.SNR if Halpha_fit is not None else np.nan for Halpha_fit in Halpha_fit_arr]
        }
    # make new table
    new_tab = Table(out_dict)
    # merge new and old tabs using filenames and write
    out_tab = join(cat, new_tab, keys_left='filepaths', keys_right='spec_filepaths')
    out_tab.remove_column("spec_filepaths")
    print(failed_fits)
    breakpoint()
    out_tab.write(prism_cat_path.replace(".fits", "_Halpha.fits"), format = "fits", overwrite = True)


def fit_Halpha(z, spec_filepath, fit_type, wav_range_AA = [6_200., 6_900.] * u.AA):
    if fit_type == "ppxf":
        fit_Halpha_ppxf(z, spec_filepath, wav_range_AA)
    elif fit_type == "manual":
        Halpha_fit = fit_Halpha_manual(z, spec_filepath, wav_range_AA)
    return Halpha_fit

def fit_Halpha_ppxf(z, spec_filepath, wav_range_AA = [6_200., 6_900.] * u.AA):
    sps_name = "fsps"

    ppxf_dir = Path(lib.__file__).parent
    basename = f"spectra_{sps_name}_9.0.npz"
    temp_filename = ppxf_dir / 'sps_models' / basename
    if not temp_filename.is_file():
        url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
        request.urlretrieve(url, temp_filename)

    spec = galfind.Spectrum.from_DJA(spec_filepath, z = z)
    wavs = spec.wavs.to(u.AA) / (1. + z)
    valid = (~spec.fluxes.mask & (wavs < wav_range_AA[1]) & (wavs > wav_range_AA[0]))
    wavs = wavs[valid]
    fluxes = spec.fluxes.filled(np.nan)[valid]
    flux_errs = spec.flux_errs.filled(np.nan)[valid]
    flux_errs = funcs.convert_mag_err_units(wavs, fluxes, flux_errs, u.erg / u.s / u.cm**2 / u.AA)[0] # symmetric in flux space
    fluxes = funcs.convert_mag_units(wavs, fluxes, u.erg / u.s / u.cm**2 / u.AA)
    norm_value = np.median(fluxes.value)
    fluxes /= norm_value
    flux_errs /= norm_value
    d_ln_wav = np.log(wavs[-1].value / wavs[0].value) / (len(wavs) - 1)  # Average ln_lam step
    velscale = const.c.to(u.km / u.s) * d_ln_wav                   # eq. (8) of Cappellari (2017)
    print(f"Velocity scale per pixel: {velscale:.2f}")
    sps = lib.sps_lib(temp_filename, velscale.value, norm_range=[5070, 5950], age_range=[0., cosmo.age(z).to(u.Gyr).value])
    
    R = 100 # PRISM
    FWHM_gal = 1e4 * np.sqrt(1.66 * 3.17) / (R * (1. + z))
    reg_dim = sps.templates.shape[1:]
    stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)
    gas_templates, gas_names, line_wave = util.emission_lines(sps.ln_lam_temp, wav_range_AA.value, FWHM_gal, tie_balmer = False)
    templates = np.column_stack([stars_templates, gas_templates])

    n_stars = stars_templates.shape[1]
    n_gas = len(gas_names)
    component = [0]*n_stars + [1]*n_gas
    gas_component = np.array(component) > 0  # gas_component=True for gas templates

    start = [1200, 200]     # (km/s), starting guess for [V, sigma]
    moments = [2, 2]
    start = [start, start]
    pp = ppxf(templates, fluxes.value, flux_errs.value, velscale.value, start,
        moments=moments, degree=-1, mdegree=-1, lam=wavs.value, lam_temp=sps.lam_temp,
        reg_dim=reg_dim, component=component, gas_component=gas_component,
        reddening=0, gas_reddening=0, gas_names=gas_names)
    plt.figure(figsize=(15, 5))
    pp.plot()
    plt.title(f"pPXF fit with {sps_name} SPS templates")
    out_path = f"Halpha_spec_fits/ppxf/{spec_filepath.split('/')[-1]}.png"
    funcs.make_dirs(out_path)
    plt.savefig(out_path)

def Halpha_gauss(x, A, sigma, c):
    return A * np.exp(-0.5 * ((x - 6562.8) / sigma)**2) + c

def Halpha_residual(params, x, y, y_err):
    # gaussian plus a constant
    model = Halpha_gauss(x, params['A'], params['sigma'], params['c'])
    return (model - y) / y_err

def Halpha_EW(x, A, sigma, c):
    return (Halpha_gauss(x, A, sigma, c) - c) / c


def fit_Halpha_manual(z, spec_filepath, wav_range_AA = [6_200., 6_900.] * u.AA):
    spec = galfind.Spectrum.from_DJA(spec_filepath, z = z)
    wavs = spec.wavs.to(u.AA) / (1. + z)
    valid = (~spec.fluxes.mask & (wavs < wav_range_AA[1]) & (wavs > wav_range_AA[0]))
    wavs = wavs[valid]
    fluxes = spec.fluxes.filled(np.nan)[valid]
    flux_errs = spec.flux_errs.filled(np.nan)[valid]
    if len(wavs) > 0:
        flux_errs = funcs.convert_mag_err_units(wavs, fluxes, flux_errs, u.erg / u.s / u.cm**2 / u.AA)[0] # symmetric in flux space
    else:
        print(f"No valid data for {spec_filepath.split('/')[-1]}")
        return None
    fluxes = funcs.convert_mag_units(wavs, fluxes, u.erg / u.s / u.cm**2 / u.AA)
    norm_factor = 1.
    fluxes *= (1. + z) ** 2
    flux_errs *= (1. + z) ** 2
    
    params = Parameters()
    params.add('A', value=np.max(fluxes.value) - np.median(fluxes.value), min=0., max=1e-12)
    params.add('c', value=np.median(fluxes.value), min=0., max=1e-12)
    params.add('sigma', value=10., min=1., max=50.)

    # dmodel = Model(gauss_model)
    # result = dmodel.fit(fluxes.value, params, wavs=wavs.value)
    # print(result.fit_report())

    out = minimize(Halpha_residual, params, args=(wavs.value,), kws={'y': fluxes.value, 'y_err': flux_errs.value})
    print(fit_report(out))

    try:
        Halpha_fit = Halpha_storage(out.params, 10_000, norm_factor, z)
    except:
        failed_fits.extend([spec_filepath])
        return None

    Halpha_EWrest_16, Halpha_EWrest_50, Halpha_EWrest_84 = Halpha_fit.get_EW_percentiles("rest")
    Halpha_fit.get_cont_percentiles()
    Halpha_fit.get_flux_percentiles()
    Halpha_fit.get_line_SNR(wavs.value, fluxes.value, flux_errs.value)

    plt.plot(wavs.value, fluxes.value, c = "black", label = "NIRSpec/PRISM")
    plt.fill_between(wavs.value, fluxes.value - flux_errs.value, fluxes.value + flux_errs.value, alpha = 0.5, color = "black")
    plt.plot(wavs.value, Halpha_fit.get_chains_median(wavs.value), c = "red", label = "Halpha model")
    model_l1, model_u1 = Halpha_fit.get_chains_errs(wavs.value)
    plt.fill_between(wavs.value, model_l1, model_u1, alpha = 0.5, color = "red")
    # make rf string containing EW width and errors
    #plt.text(0.05, 0.95, r"EW$_{\mathrm{rest}}$(H$\alpha$)=" + f"{Halpha_EWrest_50:.2f}" + r"$^{+" + f"{Halpha_EWrest_84 - Halpha_EWrest_50:.2f}" + r"}_{-" + f"{Halpha_EWrest_50 - Halpha_EWrest_16:.2f}" + r"}~\mathrm{\AA}$", transform = plt.gca().transAxes)
    # make rf string containing flux and errors
    plt.text(0.05, 0.95, r"$F_{\mathrm{H}\alpha}$=" + f"{Halpha_fit.Halpha_flux['Halpha_flux_50']:.2e}" + r"$^{+" + f"{Halpha_fit.Halpha_flux['Halpha_flux_84'] - Halpha_fit.Halpha_flux['Halpha_flux_50']:.2e}" + r"}_{-" + f"{Halpha_fit.Halpha_flux['Halpha_flux_50'] - Halpha_fit.Halpha_flux['Halpha_flux_16']:.2e}" + r"}~\mathrm{erg/s/cm^2}$", transform = plt.gca().transAxes)
    # make rf string to show the SNR
    plt.text(0.05, 0.9, f"SNR={Halpha_fit.SNR:.2f}", transform = plt.gca().transAxes)
    out_path = f"Halpha_spec_fits/manual/{spec_filepath.split('/')[-1]}.png"
    funcs.make_dirs(out_path)
    plt.xlabel("Wavelength (AA)")
    plt.ylabel("Flux (erg/s/cm^2/AA)")
    plt.legend(loc = "upper right")
    plt.savefig(out_path)
    plt.clf()
    return Halpha_fit

class Halpha_storage:

    def __init__(self, params, size, norm_factor, z):
        self.A = params["A"].value
        self.sigma = params["sigma"].value
        self.c = params["c"].value
        self.A_err = params["A"].stderr
        self.sigma_err = params["sigma"].stderr
        self.c_err = params["c"].stderr
        self.A_arr = np.random.normal(loc=self.A, scale=self.A_err, size=size)
        self.sigma_arr = np.random.normal(loc=self.sigma, scale=self.sigma_err, size=size)
        self.c_arr = np.random.normal(loc=self.c, scale=self.c_err, size=size)
        self.norm_factor = norm_factor
        self.z = z
        self.Halpha_wav = 6562.8 #* u.AA

    def get_chains(self, wavs):
        if not hasattr(self, "chains"):
            self.chains = [Halpha_gauss(wav, self.A_arr, self.sigma_arr, self.c_arr) for wav in wavs]
        return self.chains
    
    def get_chains_errs(self, wavs):
        chains = self.get_chains(wavs)
        u1 = [np.percentile(chain, 84) for chain in chains]
        l1 = [np.percentile(chain, 16) for chain in chains]
        return l1, u1
    
    def get_chains_median(self, wavs):
        chains = self.get_chains(wavs)
        return [np.percentile(chain, 50) for chain in chains]
    
    def get_chains_percentiles(self, wavs, percentiles):
        chains = self.get_chains(wavs)
        return [np.percentile(chain, percentiles) for chain in chains]
    
    def get_Halpha_flux_arr(self):
        return [self._Halpha_flux(A, sigma) * self.norm_factor for A, sigma, c in zip(self.A_arr, self.sigma_arr, self.c_arr)]
    
    @staticmethod
    def _Halpha_flux(A, sigma):
        return A * sigma * np.sqrt(2 * np.pi)
    
    def get_cont_arr(self):
        return self.c_arr * self.norm_factor
    
    def get_EW_arr(self, frame = "rest"):
        Halpha_flux_arr = self.get_Halpha_flux_arr()
        cont_arr = self.get_cont_arr()
        if frame == "rest":
            return Halpha_flux_arr / cont_arr
        elif frame == "obs":
            return Halpha_flux_arr * (1. + self.z) / cont_arr
        else:
            raise ValueError("frame must be 'rest' or 'obs'")
        
    def get_EW_percentiles(self, frame = "rest"):
        EW_arr = self.get_EW_arr(frame)
        EW_percentiles = np.percentile(EW_arr, [16, 50, 84])
        if frame == "rest":
            self.EWrest = {"EWrest_16": EW_percentiles[0], "EWrest_50": EW_percentiles[1], "EWrest_84": EW_percentiles[2]}
            return EW_percentiles[0], EW_percentiles[1], EW_percentiles[2]
        elif frame == "obs":
            self.EWobs = {"EWobs_16": EW_percentiles[0], "EWobs_50": EW_percentiles[1], "EWobs_84": EW_percentiles[2]}
            return EW_percentiles[0], EW_percentiles[1], EW_percentiles[2]
        else:
            raise(Exception("frame must be 'rest' or 'obs'"))
    
    def get_cont_percentiles(self):
        cont_arr = self.get_cont_arr()
        cont_percentiles = np.percentile(cont_arr, [16, 50, 84])
        self.cont = {"cont_16": cont_percentiles[0], "cont_50": cont_percentiles[1], "cont_84": cont_percentiles[2]}
        return cont_percentiles[0], cont_percentiles[1], cont_percentiles[2]
    
    def get_flux_percentiles(self):
        Halpha_flux_arr = self.get_Halpha_flux_arr()
        Halpha_flux_percentiles = np.percentile(Halpha_flux_arr, [16, 50, 84])
        self.Halpha_flux = {"Halpha_flux_16": Halpha_flux_percentiles[0], "Halpha_flux_50": Halpha_flux_percentiles[1], "Halpha_flux_84": Halpha_flux_percentiles[2]}
        return Halpha_flux_percentiles[0], Halpha_flux_percentiles[1], Halpha_flux_percentiles[2]
    
    def get_line_SNR(self, wavs, flux, flux_errs):
        # determine wavelength range outside of 5σ from the gaussian
        feature_mask = (wavs > self.Halpha_wav - 3. * self.sigma) & (wavs < self.Halpha_wav + 3. * self.sigma)
        SNRs = (flux[feature_mask] - self.c * self.norm_factor) / flux_errs[feature_mask]
        integrated_SNR = np.sum(SNRs) / np.sqrt(len(SNRs))
        print(f"Integrated SNR: {integrated_SNR}")
        assert not np.isnan(integrated_SNR)
        self.SNR = integrated_SNR
        return integrated_SNR

def fit_cat_MUV(survey, version, z_range, add_to_Halpha = True):
    zlabel = get_zlabel(z_range)
    prism_cat_path = get_prism_cat_path(zlabel, survey, version)
    if add_to_Halpha:
        prism_cat_path = prism_cat_path.replace(".fits", "_Halpha.fits")
    cat = Table.read(prism_cat_path)
    spec_filepaths = list(cat["filepaths"])
    z_arr = list(cat["z"])
    MUV_fit_arr = [fit_MUV(z, spec_filepath) for z, spec_filepath in tqdm(zip(z_arr, spec_filepaths), desc = "Fitting MUV", total = len(z_arr))]
    
    out_dict = {
        "spec_filepaths": spec_filepaths,
        "MUV": [MUV_fit.MUV if MUV_fit is not None else np.nan for MUV_fit in MUV_fit_arr], \
        "MUV_l1": [MUV_fit.MUV_l1 if MUV_fit is not None else np.nan for MUV_fit in MUV_fit_arr], \
        "MUV_u1": [MUV_fit.MUV_u1 if MUV_fit is not None else np.nan for MUV_fit in MUV_fit_arr], \
        }
    # make new table
    new_tab = Table(out_dict)
    # merge new and old tabs using filenames and write
    out_tab = join(cat, new_tab, keys_left='filepaths', keys_right='spec_filepaths')
    out_tab.remove_column("spec_filepaths")
    print(failed_fits)
    breakpoint()
    out_tab.write(prism_cat_path.replace(".fits", "_MUV.fits"), format = "fits", overwrite = True)

class MUV_storage:
    def __init__(self, flambda_1500, flambda_1500_err, z, size, norm_factor = 1.):
        self.flambda_1500 = flambda_1500
        self.flambda_1500_err = flambda_1500_err
        self.z = z
        self.size = size
        self.norm_factor = norm_factor
        self._calc_MUV()
    
    def get_flambda_1500_chains(self):
        return np.random.normal(self.flambda_1500.value, self.flambda_1500_err.value, self.size)

    def _calc_MUV(self):
        flambda = self.get_flambda_1500_chains() * self.norm_factor * u.erg / (u.s * (u.cm ** 2) * u.AA)
        fnu = galfind.useful_funcs_austind.convert_mag_units(1500. * u.AA, flambda, u.Jy)
        mUV = -2.5 * np.log10(fnu.value) + 8.9
        mUV += 2.5 * np.log10(self.norm_factor)
        MUV = mUV - 5. * np.log10(funcs.calc_lum_distance(self.z).to(u.pc).value / 10.) + 2.5 * np.log10(1. + self.z)
        self.MUV = np.nanmedian(MUV)
        self.MUV_l1 = self.MUV - np.nanpercentile(MUV, 16)
        self.MUV_u1 = np.nanpercentile(MUV, 84) - self.MUV
        #print(self.MUV, self.MUV_l1, self.MUV_u1)

class xi_ion_storage:
    def __init__(self, Halpha_flux_chain, MUV_chain):
        self.Halpha_flux_chain = Halpha_flux_chain
        self.MUV_chain = MUV_chain

def fit_MUV(z, spec_filepath, wav_range_AA = [1_450., 1_550.] * u.AA, size = 10_000):
    spec = galfind.Spectrum.from_DJA(spec_filepath, z = z)
    wavs = spec.wavs.to(u.AA) / (1. + z)
    valid = (~spec.fluxes.mask & (wavs < wav_range_AA[1]) & (wavs > wav_range_AA[0]))
    wavs = wavs[valid]
    fluxes = spec.fluxes.filled(np.nan)[valid]
    flux_errs = spec.flux_errs.filled(np.nan)[valid]
    if len(wavs) > 0:
        try:
            flux_errs = funcs.convert_mag_err_units(wavs, fluxes, flux_errs, u.erg / u.s / u.cm**2 / u.AA)[0] # symmetric in flux space
        except Exception as e:
            print(f"Failed to convert mag err units for {spec_filepath.split('/')[-1]}")
            print(e)
            return None
    else:
        print(f"No valid data for {spec_filepath.split('/')[-1]}")
        return None
    fluxes = funcs.convert_mag_units(wavs, fluxes, u.erg / u.s / u.cm**2 / u.AA)
    fluxes *= (1. + z) ** 2
    flux_errs *= (1. + z) ** 2
    # weighted mean
    weights = flux_errs ** -2.
    flambda_1500 = np.sum(fluxes * weights) / (np.sum(weights) * len(fluxes))
    flambda_1500_err = np.sqrt(1. / np.sum(weights))
    if flambda_1500 < 0.:
        return None
    else:
        norm_factor = 100. * flambda_1500.value
    #print(flambda_1500, flambda_1500_err)
    MUV_fit = MUV_storage(flambda_1500, flambda_1500_err, z, size = size, norm_factor = norm_factor)
    return MUV_fit


def plot(survey, version, z_range):
    # plot using galfind stylesheet
    plt.style.use(f"{galfind.config['DEFAULT']['GALFIND_DIR']}/galfind_style.mplstyle")

    zlabel = get_zlabel(z_range)
    prism_cat_path = get_prism_cat_path(zlabel, survey, version)
    Halpha_cat_path = prism_cat_path.replace(".fits", "_Halpha.fits")
    cat = Table.read(Halpha_cat_path)
    # plot log10(flux) vs z
    Ha_flux_nans = np.isnan(cat["Ha_flux"].filled().value)
    Ha_flux = cat["Ha_flux"].filled().value[~Ha_flux_nans]
    Ha_flux_l1 = cat["Ha_flux_l1"].filled().value[~Ha_flux_nans]
    Ha_flux_u1 = cat["Ha_flux_l1"].filled().value[~Ha_flux_nans]
    z = cat["z"][~Ha_flux_nans]
    Ha_flux_l1 = np.log10(Ha_flux) - np.log10(Ha_flux - Ha_flux_l1)
    Ha_flux_u1 = np.log10(Ha_flux + Ha_flux_u1) - np.log10(Ha_flux)
    Ha_flux = np.log10(Ha_flux)
    print(list(cat["filenames"][~Ha_flux_nans][np.isnan(Ha_flux_l1)]))
    print(list(cat["filenames"][~Ha_flux_nans][np.isnan(Ha_flux_u1)]))
    # mask out nans
    mask = ~np.isnan(Ha_flux)
    z = z[mask]
    Ha_flux = Ha_flux[mask]
    Ha_flux_l1 = Ha_flux_l1[mask]
    Ha_flux_u1 = Ha_flux_u1[mask]
    plt.scatter(z, Ha_flux, zorder = 10, c = cat["Ha_SNR"][~Ha_flux_nans][mask], cmap = "viridis")
    # plot slightly transparent errorbars
    plt.errorbar(z, Ha_flux, yerr=[Ha_flux_l1, Ha_flux_u1], fmt='', markeredgecolor='black', ecolor='black', linestyle='None', alpha = 0.5)
    plt.xlabel("Redshift, z")
    plt.ylabel(r"$\log_{10}$F(H$\alpha$) / erg/s/cm^2")
    plt.colorbar(label = "SNR")
    save_path = f"plots/spec_Halpha_flux_vs_z_{zlabel}.png"
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok = True)
    plt.savefig(save_path)


no_data_dict = {"6.5<z<8.5": [
    "gds-barrufet-s156-v3_prism-clear_2198_3305.spec.fits", 
    "rubies-uds41-nod-v3_prism-clear_4233_40505.spec.fits", # doubled with rubies-uds41-v3_prism-clear_4233_40505.spec.fits
    ]}

dodgy_reduction = {"6.5<z<8.5": [
    "rubies-uds2-v3_prism-clear_4233_59492.spec.fits",
    "rubies-uds2-nod-v3_prism-clear_4233_59492.spec.fits"
    ]}

too_faint = {"6.5<z<8.5": ["rubies-uds1-v3_prism-clear_4233_-7.spec.fits"]}

def load_phot():
    pass

def load_phot_errs():
    pass

def main():
    fit_cat_MUV("JOF", "v11", [5.6, 6.5], add_to_Halpha = True)
    #fit_cat_Halpha("JOF", "v11", [5.6, 6.5], "manual")
    #plot("JOF", "v11", [5.6, 6.5])
    
    #make_prism_cat()
    #plot_prism_data("JOF", "v11", [5.6, 6.5])
    #write_high_res_Halpha_tab()
    

if __name__ == "__main__":
    main()
