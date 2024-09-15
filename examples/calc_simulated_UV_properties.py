#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:03:27 2023

@author: austind
"""

# calc_simulated_beta.py
import astropy.units as u
import numpy as np
from astropy.table import Table, join
from tqdm import tqdm

from galfind import (
    ACS_WFC,
    Instrument,
    NIRCam,
    Photometry,
    Photometry_rest,
    config,
    useful_funcs_austind,
)


def jaguar_phot_keys(band):
    if band in NIRCam().bands:
        return f"NRC_{band.replace('f', 'F')}_fnu"
    elif band in ACS_WFC().bands:
        return f"HST_{band.replace('f', 'F')}_fnu"


def jaguar_phot_unit_conv(flux_nu):
    return (np.array(flux_nu) * u.nJy).to(u.Jy).value


def main(
    cat_path, sample_key, instrument_name, templates="fsps_larson", save=True
):
    # load catalogue
    tab = Table.read(cat_path, character_as_bytes=False)
    tab = tab[tab[sample_key] == True]
    IDs = np.array(tab["NUMBER"])
    # RA = np.array(tab["ALPHA_J2000"])
    # DEC = np.array(tab["DELTA_J2000"])
    true_redshifts = np.array(tab["redshift"])
    obs_redshifts = np.array(tab[f"zbest_{templates}"])

    instrument = Instrument.from_name(instrument_name)
    for band in instrument:
        try:
            tab[f"MAG_APER_{band.band_name}"]
        except:
            instrument -= band
            pass
    print(instrument.band_names)

    beta_raw_arr = []
    m_UV_raw_arr = []
    M_UV_raw_arr = []
    beta_scattered_arr = []
    m_UV_scattered_arr = []
    M_UV_scattered_arr = []
    beta_scattered_arr_zfix = []
    m_UV_scattered_arr_zfix = []
    M_UV_scattered_arr_zfix = []
    # obs_rest_UV_band_arr = []
    # obs_rest_UV_mag_arr = []
    for true_z, photo_z, row in tqdm(
        zip(true_redshifts, obs_redshifts, tab),
        total=len(true_redshifts),
        desc=f"Calculating beta/M_UV for {survey} {templates} JAGUAR catalogue",
    ):
        # calculate beta/M_UV for raw photometry
        raw_phot_vals = [
            jaguar_phot_unit_conv(row[jaguar_phot_keys(band_name)])
            for band_name in instrument.band_names
        ]
        # run raw UV properties through galfind
        raw_phot_obj = Photometry(
            instrument,
            raw_phot_vals * u.Jy,
            np.full(len(raw_phot_vals), 0.0 * u.Jy),
            [],
        )
        raw_phot_rest_obj = Photometry_rest.from_phot(
            raw_phot_obj, float(true_z)
        )
        raw_beta = raw_phot_rest_obj.basic_beta_calc(incl_errs=False)
        raw_m_UV = raw_phot_rest_obj.basic_m_UV_calc(1.0, incl_errs=False)
        raw_M_UV = raw_phot_rest_obj.basic_M_UV_calc(1.0, incl_errs=False)
        beta_raw_arr.append(raw_beta)
        m_UV_raw_arr.append(raw_m_UV)
        M_UV_raw_arr.append(raw_M_UV)

        # calculate beta/M_UV for scattered photometry
        scattered_phot_vals = [
            jaguar_phot_unit_conv(row[f"FLUX_APER_{band_name}"][0])
            for band_name in instrument.band_names
        ]
        scattered_phot_errs = [
            jaguar_phot_unit_conv(row[f"FLUXERR_APER_{band_name}"][0])
            for band_name in instrument.band_names
        ]

        # run scattered UV properties through galfind
        scattered_phot_obj = Photometry(
            instrument,
            scattered_phot_vals * u.Jy,
            scattered_phot_errs * u.Jy,
            [],
        )

        # z fixed + scattered
        scattered_phot_rest_obj_zfix = Photometry_rest.from_phot(
            scattered_phot_obj, float(true_z)
        )
        scattered_beta_zfix = scattered_phot_rest_obj_zfix.basic_beta_calc(
            incl_errs=True
        )
        scattered_m_UV_zfix = scattered_phot_rest_obj_zfix.basic_m_UV_calc(
            1.0, incl_errs=True
        )
        scattered_M_UV_zfix = scattered_phot_rest_obj_zfix.basic_M_UV_calc(
            1.0, incl_errs=True
        )
        # obs_rest_UV_band = scattered_phot_rest_obj_zfix.rest_UV_band
        # obs_rest_UV_band_arr.append(obs_rest_UV_band)
        # obs_rest_UV_mag_arr.append(float(row[f"MAG_APER_{obs_rest_UV_band}"][0]))
        beta_scattered_arr_zfix.append(scattered_beta_zfix)
        m_UV_scattered_arr_zfix.append(scattered_m_UV_zfix)
        M_UV_scattered_arr_zfix.append(scattered_M_UV_zfix)

        # z measured + scattered
        scattered_phot_rest_obj = Photometry_rest.from_phot(
            scattered_phot_obj, float(photo_z)
        )
        scattered_beta = scattered_phot_rest_obj.basic_beta_calc(
            incl_errs=True
        )
        scattered_m_UV = scattered_phot_rest_obj.basic_m_UV_calc(
            1.0, incl_errs=True
        )
        scattered_M_UV = scattered_phot_rest_obj.basic_M_UV_calc(
            1.0, incl_errs=True
        )
        beta_scattered_arr.append(scattered_beta)
        m_UV_scattered_arr.append(scattered_m_UV)
        M_UV_scattered_arr.append(scattered_M_UV)

    # save table
    if save:
        save_name = f"{config['DEFAULT']['GALFIND_WORK']}/Catalogues/simulated/{instrument_name}/{survey}-Jaguar/{cat_path.split('/')[-1].replace('.fits', '_UV.fits')}"
        useful_funcs_austind.make_dirs(save_name)
        UV_tab = Table(
            [
                IDs,
                beta_raw_arr,
                m_UV_raw_arr,
                M_UV_raw_arr,
                beta_scattered_arr,
                m_UV_scattered_arr,
                M_UV_scattered_arr,
                beta_scattered_arr_zfix,
                m_UV_scattered_arr_zfix,
                M_UV_scattered_arr_zfix,
            ],
            names=(
                "ID",
                "Beta_true",
                "m_UV_true",
                "M_UV_true",
                f"Beta_EAZY_{templates}",
                f"m_UV_EAZY_{templates}",
                f"M_UV_EAZY_{templates}",
                f"Beta_EAZY_{templates}_zfix",
                f"m_UV_EAZY_{templates}_zfix",
                f"M_UV_EAZY_{templates}_zfix",
            ),
            dtype=(
                int,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
            ),
        )
        out_tab = join(tab, UV_tab, keys_left="NUMBER", keys_right="ID")
        # save table
        out_tab.write(save_name, overwrite=True)
        print(f"Saved beta/M_UV for JAGUAR using {templates} to {save_name}")


if __name__ == "__main__":
    surveys = [
        "CEERS",
        "CLIO",
        "El-Gordo",
        "GLASS",
        "MACS-0416",
        "NEP",
        "NGDEEP",
        "SMACS-0723",
        "JADES",
    ]
    instrument_name = "ACS_WFC+NIRCam"
    version = "v9"
    templates = "fsps_larson"
    min_pc_err = 10
    save = True

    for survey in surveys:
        cat_path = f"/raid/scratch/work/austind/GALFIND_WORK/Catalogues/simulated/NIRCam+ACS_WFC+WFC3_IR/{survey}-Jaguar/JAGUAR_SimDepth_{survey}_{version}_half_{str(int(min_pc_err))}pc_EAZY_matched_selection.fits"
        sample_key = f"final_sample_highz_{templates}"
        main(
            cat_path,
            sample_key,
            instrument_name,
            templates=templates,
            save=save,
        )
