
import galfind

from astropy.table import Table
from tqdm import tqdm
import numpy as np

def write_high_res_Halpha_tab():
    spec_cat = galfind.Spectral_Catalogue.from_DJA(
        version = "v3", z_cat_range = [5.5, 6.5], grade = 3
    )
    #medium_res = [spec.meta["FILENAME"] for spec_arr in spec_cat for spec in spec_arr if spec.instrument.grating.name == "G395M"]
    high_res = [spec.origin.split("/")[-1] for spec_arr in spec_cat for spec in spec_arr if spec.instrument.grating.name == "G395H"]
    grating_names = ["G395H" for spec in high_res]
    #grating_names.extend(["G395H" for spec in high_res])
    filenames = high_res
    #filenames.extend(high_res)
    tab = Table({"grating": grating_names, "filename": filenames})
    tab.write("G395H,55z65.ecsv", format = "ascii", overwrite  = True)

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

def main():

    survey = "JOF"
    version = "v11"
    depth_region = "all"
    z_range = [5.6, 6.5]
    z_label = f"{str(z_range[0]).replace('.', '_')}<z<{str(z_range[1]).replace('.', '_')}"
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
    #breakpoint()
    tab = Table({"ID": ID, "z": z, "filenames": filenames, **phot_dict, **scattered_phot_dict, **phot_errs_dict})
    tab.write(f"PRISM,{z_label}_{survey}_{version}.fits", format = "ascii", overwrite  = True)

if __name__ == "__main__":
    main()
