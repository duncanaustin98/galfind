
import pytest
import matplotlib.pyplot as plt

from galfind import Galaxy

# @pytest.mark.requires_data
# def test_gal_lephare_loaded(gal_lephare_loaded):
#     assert isinstance(gal_lephare_loaded, Galaxy)

@pytest.fixture(
    scope = "module",
    params = [
        {
            "overwrite": True,
        },
    ]
)
def plot_phot_diagnostic_kwargs(request):
    return request.param

# TODO: Generalize to run with all sed fitters
@pytest.mark.requires_data
def test_gal_lephare_loaded_plot_phot_diagnostic(
    gal_custom_lephare_loaded,
    custom_lephare_sed_fitter,
    data,
    plot_phot_diagnostic_kwargs,
):
    #fig, ax = plt.subplots()
    gal_custom_lephare_loaded.plot_phot_diagnostic(
        data,
        SED_arr = custom_lephare_sed_fitter,
        zPDF_arr = custom_lephare_sed_fitter,
        **plot_phot_diagnostic_kwargs
    )

@pytest.mark.requires_data
def test_gal_lephare_eazy_plot_phot_diagnostic(
    gal_custom_lephare_eazy_loaded,
    custom_lephare_sed_fitter,
    eazy_fsps_larson_sed_fitter,
    data,
    plot_phot_diagnostic_kwargs,
):
    #fig, ax = plt.subplots()
    gal_custom_lephare_eazy_loaded.plot_phot_diagnostic(
        data,
        SED_arr = [custom_lephare_sed_fitter, eazy_fsps_larson_sed_fitter],
        zPDF_arr = [custom_lephare_sed_fitter, eazy_fsps_larson_sed_fitter],
        **plot_phot_diagnostic_kwargs
    )
