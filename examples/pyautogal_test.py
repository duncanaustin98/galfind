
import time
from os import path
import numpy as np
from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt
start = time.time()
import autogalaxy as ag
import autogalaxy.plot as aplt
import autofit as af
end = time.time()
print("pyautogal import time: ", end - start)
from galfind import config

def main():
    ID = 288
    band = "F444W"
    cutout_path = f"/{config['DEFAULT']['GALFIND_WORK']}/Cutouts/v11/JOF/0.96as/{band}/data/{str(ID)}.fits"
    PSF_path = f"/{config['DEFAULT']['GALFIND_DATA']}/jwst/PSFs/PSF_Resample_03_{band}.fits"

    dataset = ag.Imaging.from_fits(
        data_path = cutout_path,
        data_hdu = 1,
        noise_map_path = cutout_path,
        noise_map_hdu = 3,
        psf_path = PSF_path,
        pixel_scales = 0.03,
    )
    breakpoint()
    mask = ag.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=0.01
    )
    dataset = dataset.apply_mask(mask=mask)

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

    galaxy_model = af.Model(ag.Galaxy, redshift=0.5, bulge=ag.lp_linear.Sersic)
    model = af.Collection(galaxies=af.Collection(galaxy=galaxy_model))
    print(model.info)
    analysis = ag.AnalysisImaging(dataset=dataset)
    search = af.LBFGS()
    result = search.fit(model=model, analysis=analysis)
    print(result.info)
    breakpoint()

if __name__ == "__main__":
    main()