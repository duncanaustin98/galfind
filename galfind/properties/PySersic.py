"""PySersic-based morphological fitting with straightforward prior control.

Provides `PySersic_Fitter`/`PySersic_Result`, a `pysersic`-backed
alternative to `Galfit_Fitter`/`Galfit_Result` (see `Morphology.py`).
Unlike Galfit's text-file box constraints, priors here are genuine
Bayesian distributions fit with `pysersic`'s SVI machinery, but the user
interface mirrors Galfit's `primary_constraints`/`neighbour_constraints`
dicts closely: a `[lo, hi]` value means a flat/uniform prior over that
range, and a single scalar means the parameter is held fixed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import astropy.units as u
import h5py
import numpy as np
from astropy.nddata import Cutout2D
from astropy.table import Table
from tqdm import tqdm

if TYPE_CHECKING:
    from . import Band_Cutout_Base, Catalogue, Filter, PSF_Base

try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from .. import config
from ..utils import useful_funcs_austind as funcs
from ..utils.exceptions import (
    GalfindTypeError,
    IncompatibleKwargsError,
    InvalidOptionError,
    LengthMismatchError,
    RangeError,
)
from ..visualization.PDF import PDF
from .Morphology import Morphology_Fitter, Morphology_Result

PriorSpec = Dict[str, Dict[str, Union[int, float, List[Union[int, float]]]]]


class PySersic_Result(Morphology_Result):
    """Results of a `pysersic` morphological fit to a single source.

    Parameters
    ----------
    fitter : `PySersic_Fitter`
        The `PySersic_Fitter` instance that produced this result.
    chi2 : `float`
        Chi-squared of the fit over unmasked pixels.
    Ndof : `int`
        Number of degrees of freedom of the fit.
    properties : `dict` of `str` to `astropy.units.Quantity`
        Best-fit (median posterior) primary-source properties.
    property_errs : `dict` of `str` to `list`
        16th/84th-percentile-derived lower/upper uncertainties on each
        entry of `properties`.
    property_pdfs : `dict` of `str` to `PDF`
        Real posterior distributions (not approximated), one per
        primary-source property, built from `pysersic`'s SVI draws.
    model_image : `numpy.ndarray`, optional
        Median posterior model image. Default is `None`.
    neighbour_properties : `list` of `dict`, optional
        Per-neighbour best-fit properties, in the same order neighbours
        were passed to the fit. Empty for a single-source fit. Default
        is `None` (stored as `[]`).
    param_samples : `dict` of `str` to `numpy.ndarray`, optional
        Flattened posterior draws per primary-source parameter (same
        keys as `properties`), used to build the corner plot in
        `plot()`. Default is `None` (stored as `{}`, corner plot
        skipped).
    rff : `float`, optional
        Residual Flux Fraction of the fit. Default is `None`.
    cutout : `Type[Band_Cutout_Base]`, optional
        Cutout of the source that was fitted. Default is `None`.
    """

    def __init__(
        self: Self,
        fitter: "PySersic_Fitter",
        chi2: float,
        Ndof: int,
        properties: Dict[str, u.Quantity],
        property_errs: Dict[str, List[u.Quantity]],
        property_pdfs: Dict[str, PDF],
        model_image: Optional[np.ndarray] = None,
        neighbour_properties: Optional[List[Dict[str, Any]]] = None,
        param_samples: Optional[Dict[str, np.ndarray]] = None,
        rff: Optional[float] = None,
        cutout: Optional[Type[Band_Cutout_Base]] = None,
    ) -> None:
        self.model_image = model_image
        self.neighbour_properties = neighbour_properties or []
        self.param_samples = param_samples or {}
        super().__init__(
            fitter,
            chi2,
            Ndof,
            properties,
            property_errs,
            property_pdfs,
            rff=rff,
            cutout=cutout,
        )

    def plot(
        self: Self,
        save: bool = True,
        show: bool = False,
        out_path: Optional[str] = None,
    ) -> None:
        """Plot the fit's science/model/residual images and posterior
        corner plot.

        Parameters
        ----------
        save : `bool`, optional
            Whether to save the figures to disk. Default is `True`.
        show : `bool`, optional
            Whether to display the figures. Default is `False`.
        out_path : `str`, optional
            Base path (without extension) to save the residual plot to;
            the corner plot is saved alongside it with a `_corner`
            suffix. Required if `save` is `True`.

        Raises
        ------
        IncompatibleKwargsError
            If `save` is `True` but `out_path` is `None`.
        """
        import matplotlib.pyplot as plt
        from pysersic.results import plot_residual

        sci = self.cutout.load("SCI")[1] if self.cutout is not None else None
        mask = getattr(self, "_fit_mask", None)
        if sci is not None and self.model_image is not None:
            residual_std = float(
                np.std((sci - self.model_image)[~mask])
                if mask is not None
                else np.std(sci - self.model_image)
            )
            fig, ax = plot_residual(
                sci,
                self.model_image,
                mask=mask,
                vmin=-3 * residual_std,
                vmax=3 * residual_std,
            )
            if save:
                if out_path is None:
                    raise IncompatibleKwargsError(
                        "out_path must be given when save=True in "
                        "PySersic_Result.plot()."
                    )
                funcs.make_dirs(out_path)
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(fig)

        if self.param_samples:
            import corner as corner_pkg

            labels = list(self.param_samples.keys())
            samples = np.column_stack(
                [self.param_samples[name] for name in labels]
            )
            corner_fig = corner_pkg.corner(
                samples,
                labels=labels,
                range=PySersic_Fitter._corner_ranges(self.param_samples),
                show_titles=True,
                quantiles=[0.16, 0.5, 0.84],
            )
            if save:
                if out_path is None:
                    raise IncompatibleKwargsError(
                        "out_path must be given when save=True in "
                        "PySersic_Result.plot() (corner plot)."
                    )
                corner_path = str(
                    Path(out_path).with_name(
                        f"{Path(out_path).stem}_corner{Path(out_path).suffix}"
                    )
                )
                funcs.make_dirs(corner_path)
                corner_fig.savefig(corner_path, dpi=150, bbox_inches="tight")
            if show:
                plt.show()
            else:
                plt.close(corner_fig)


class PySersic_Fitter(Morphology_Fitter):
    """Morphological fitting using `pysersic`.

    Fits Sersic and/or point-source profiles to galaxy images with
    `pysersic`'s SVI machinery, with straightforward prior control and
    (optionally) simultaneous fitting of neighbouring sources -- the
    `pysersic` analogue of `Galfit_Fitter`.

    Parameters
    ----------
    psf : `Type[PSF_Base]`
        PSF model for convolution.
    model : `str`, optional
        Profile to fit the primary source with -- one of `"sersic"`
        (pure Sersic host), `"pointsource"` (pure unresolved point
        source), or `"sersic_pointsource"` (host + nuclear point
        source). Default is `"sersic"`.
    primary_priors : `dict`, optional
        Prior overrides for the primary source, keyed by `model` then by
        `pysersic` parameter name (e.g. `"n"`, `"r_eff"`, `"ellip"`,
        `"theta"`, `"f_ps"`). A 2-element `[lo, hi]` list replaces
        `pysersic`'s default prior for that parameter with a flat/uniform
        prior over that range; a single `int`/`float` holds the
        parameter fixed at that value. `xc`/`yc` are not settable here --
        see `xy_center_sigma`. Example:
        ``{"sersic": {"n": 1.0, "r_eff": [0.5, 40.0]}}``. Default is
        `None` (every parameter fitted with `pysersic`'s own defaults).
    neighbour_priors : `dict`, optional
        Same format as `primary_priors`, applied to every detected
        neighbour when `neighbours_model` is set. Default is `None`.
    neighbours_model : `str`, optional
        If given, any neighbouring sources found via `_get_neighbours`
        are fit *simultaneously* with the primary source (using
        `pysersic.FitMulti`) with this profile type. If `None` (default),
        only the primary source is fit (`pysersic.FitSingle`).
    xy_center_sigma : `float`, optional
        Standard deviation (pixels) of the Gaussian prior on every
        source's center. Default is `0.5`.
    rkey_seed : `int`, optional
        Seed for `pysersic`'s `estimate_posterior` JAX rng key. Default
        is `3`.
    method : `str`, optional
        `pysersic` `estimate_posterior` method (`"svi-flow"`, `"svi-mvn"`,
        or `"laplace"`). Default is `"svi-flow"`.

    Attributes
    ----------
    primary_fixed, primary_bounded : `dict`
        Parsed fixed-value / `[lo, hi]`-bounded overrides for the primary
        source, keyed by parameter name.
    neighbour_fixed, neighbour_bounded : `dict`
        Same, for neighbours.
    """

    def __init__(
        self: Self,
        psf: Type[PSF_Base],
        model: str = "sersic",
        primary_priors: Optional[PriorSpec] = None,
        neighbour_priors: Optional[PriorSpec] = None,
        neighbours_model: Optional[str] = None,
        xy_center_sigma: float = 0.5,
        rkey_seed: int = 3,
        method: str = "svi-flow",
    ) -> None:
        self.primary_fixed, self.primary_bounded = self._load_priors(
            primary_priors, model
        )
        self.neighbours_model = neighbours_model
        if neighbours_model is not None:
            self.neighbour_fixed, self.neighbour_bounded = self._load_priors(
                neighbour_priors, neighbours_model
            )
        else:
            self.neighbour_fixed, self.neighbour_bounded = {}, {}
        self.xy_center_sigma = xy_center_sigma
        self.rkey_seed = rkey_seed
        self.method = method
        super().__init__(psf, model)

    @property
    def _available_models(self: Self) -> List[str]:
        # pysersic's own profile_type strings, used directly -- no
        # renaming/translation layer.
        return ["sersic", "pointsource", "sersic_pointsource"]

    @staticmethod
    def _load_priors(
        priors: Optional[PriorSpec], model: Optional[str]
    ) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
        """Split a `{model: {param: value_or_[lo,hi]}}` spec into
        fixed-value and `[lo, hi]`-bounded dicts for a single `model`.

        Raises
        ------
        InvalidOptionError
            If `model` is not a valid `pysersic` profile type, or if a
            parameter name in `priors[model]` is not valid for `model`.
        LengthMismatchError
            If a `[lo, hi]`-style prior bound does not have exactly 2
            elements.
        RangeError
            If a `[lo, hi]`-style prior bound does not satisfy `lo < hi`.
        GalfindTypeError
            If a prior value is neither a `[lo, hi]` list/tuple nor a
            single `int`/`float`.
        """
        from pysersic.priors import base_profile_params

        fixed: Dict[str, float] = {}
        bounded: Dict[str, Tuple[float, float]] = {}
        if not priors or model is None or model not in priors:
            return fixed, bounded
        if model not in base_profile_params:
            raise InvalidOptionError(
                f"model={model!r} is not a valid pysersic profile_type; "
                f"must be one of {sorted(base_profile_params.keys())}."
            )
        valid_params = set(base_profile_params[model])
        for param, val in priors[model].items():
            if param not in valid_params:
                raise InvalidOptionError(
                    f"param={param!r} is not a valid parameter for "
                    f"model={model!r}; must be one of "
                    f"{sorted(valid_params)}."
                )
            if isinstance(val, (list, tuple)):
                if len(val) != 2:
                    raise LengthMismatchError(
                        f"Prior bound for param={param!r} must be a "
                        f"[lo, hi] pair of length 2; got val={val!r} "
                        f"with length {len(val)}."
                    )
                lo, hi = float(val[0]), float(val[1])
                if not lo < hi:
                    raise RangeError(
                        f"Prior bound for param={param!r} must have "
                        f"lo < hi; got val={val!r}."
                    )
                bounded[param] = (lo, hi)
            elif isinstance(val, (int, float)):
                fixed[param] = float(val)
            else:
                raise GalfindTypeError(
                    f"Invalid prior for param={param!r} in model={model!r}; "
                    f"must be a [lo, hi] list or a single int/float. Got "
                    f"val={val!r} with type {type(val).__name__}."
                )
        return fixed, bounded

    @staticmethod
    def _apply_prior_overrides(
        prior: Any,
        fixed: Dict[str, float],
        bounded: Dict[str, Tuple[float, float]],
        suffix: str = "",
    ) -> None:
        """Apply parsed fixed/bounded overrides onto a `pysersic` prior
        object in place (`suffix` selects a source index for multi-source
        priors, e.g. `"_0"` for the primary, `"_1"` for the first
        neighbour)."""
        for param, (lo, hi) in bounded.items():
            prior.set_uniform_prior(param + suffix, lo, hi)
        for param in fixed.keys():
            key = param + suffix
            prior.dist_dict.pop(key, None)
            prior.reparam_dict.pop(key, None)
            prior.repr_dict.pop(param, None)

    @staticmethod
    def _patch_fixed_params_build_model(
        fitter: Any, fixed_params_by_key: Dict[str, float], multi: bool = False
    ) -> None:
        """Monkeypatch a `pysersic` `FitSingle`/`FitMulti` instance so
        the parameters named in `fixed_params_by_key` (already-suffixed
        keys, e.g. `"n"` or `"n_0"`) are true constants rather than
        fitted parameters.

        There is no supported `pysersic` API for holding a parameter
        fixed while fitting the rest: `BasePrior.__call__` unconditionally
        calls `numpyro.sample()` on every entry of `prior.dist_dict`, and
        numpyro's SVI/MCMC initialization explicitly rejects a bare
        `sample` site backed by a `dist.Delta` ("Sample site '...' has a
        delta distribution; use numpyro.deterministic to add this value
        to the trace instead") -- so a fixed parameter can't be sampled
        at all. The caller must already have removed every key in
        `fixed_params_by_key` from `prior.dist_dict` (and
        `reparam_dict`/`repr_dict`) via `_apply_prior_overrides`, so
        `prior()` no longer tries to sample it; this then injects each
        one back into the per-draw params dict as a `numpyro.deterministic`
        constant before rendering, so it's fixed for every posterior draw
        while still showing up (as a zero-width column) in
        `svi_results`/corner plots. Pinned to `pysersic.pysersic`'s
        `FitSingle`/`FitMulti.build_model` (1.x); would need revisiting
        if those change.
        """
        import jax.numpy as jnp
        import numpyro
        from numpyro import deterministic

        prior = fitter.prior

        def build_model(return_model: bool = True):
            @numpyro.handlers.reparam(config=prior.reparam_dict)
            def model(return_model: bool = return_model):
                params = prior()
                for key, val in fixed_params_by_key.items():
                    params[key] = deterministic(key, jnp.asarray(float(val)))
                if multi:
                    out = fitter.renderer.render_for_model(
                        params, prior.catalog["type"], suffix=prior.suffix
                    )
                else:
                    out = fitter.renderer.render_source(
                        params, prior.profile_type, suffix=prior.suffix
                    )
                sky = prior.sample_sky(fitter.renderer.X, fitter.renderer.Y)
                obs = out + sky
                if return_model:
                    deterministic(f"model{prior.suffix}", obs)
                fitter.loss_func(
                    obs,
                    fitter.data,
                    fitter.rms,
                    fitter.mask,
                    suffix=prior.suffix,
                )

            return model

        fitter.build_model = build_model

    @staticmethod
    def _corner_ranges(samples: Dict[str, np.ndarray]) -> List[Any]:
        """Per-column `range` list for the `corner` package. A truly
        fixed parameter has exactly zero posterior variance, which
        `corner` rejects outright unless given an explicit `range`; pad
        any such column, leave normally-varying columns on corner's own
        default (`1.0`, i.e. the full data range)."""
        ranges: List[Any] = []
        for arr in samples.values():
            arr = np.asarray(arr)
            lo, hi = float(np.min(arr)), float(np.max(arr))
            ranges.append((lo - 0.5, hi + 0.5) if hi - lo < 1e-8 else 1.0)
        return ranges

    def _fit_array(
        self: Self,
        sci: np.ndarray,
        rms: np.ndarray,
        mask: np.ndarray,
        psf_arr: np.ndarray,
        neighbour_catalog: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[Dict[str, Any], Any]:
        """Core array-level fit -- no galfind `Cutout`/`Data`/`Galaxy`
        dependency, so it's directly unit-testable with synthetic
        arrays.

        Parameters
        ----------
        sci, rms, mask, psf_arr : `numpy.ndarray`
            Science, RMS-error, boolean mask (`True` = masked), and PSF
            (same shape as `sci`) arrays.
        neighbour_catalog : `dict`, optional
            `{"x", "y", "flux", "r", "type"}` arrays (pixel coordinates
            in `sci`), primary source at index 0, one or more neighbours
            after it. If given, fits all sources simultaneously with
            `pysersic.FitMulti`; if `None` (default), fits only the
            primary source with `pysersic.FitSingle`.

        Returns
        -------
        `tuple`
            `(map_params, idata)`, where `map_params` is `pysersic`'s
            16/50/84th-percentile quantile dict (plus `"model"`, the
            median posterior model image) and `idata` is the full
            `arviz` `InferenceData` object.
        """
        import jax
        from pysersic import FitMulti, FitSingle, check_input_data
        from pysersic.priors import PySersicMultiPrior, SourceProperties

        check_input_data(data=sci, rms=rms, psf=psf_arr, mask=mask)

        if neighbour_catalog is None:
            props = SourceProperties(sci, mask=mask)
            center_y, center_x = np.array(sci.shape) // 2
            props.set_position_guess((center_x, center_y))
            if not np.isfinite(props.r_eff_guess):
                props.set_r_eff_guess(r_eff_guess=3.0, r_eff_guess_err=3.0)
            prior = props.generate_prior(self.model, sky_type="flat")
            self._apply_prior_overrides(
                prior, self.primary_fixed, self.primary_bounded
            )
            if "xc" in prior.dist_dict:
                prior.set_gaussian_prior(
                    "xc", props.xc_guess, self.xy_center_sigma
                )
                prior.set_gaussian_prior(
                    "yc", props.yc_guess, self.xy_center_sigma
                )
            fitter = FitSingle(
                data=sci,
                rms=rms,
                mask=mask,
                psf=psf_arr,
                prior=prior,
                renderer_kwargs={"use_interp_amps": True},
            )
            if self.primary_fixed:
                self._patch_fixed_params_build_model(
                    fitter, dict(self.primary_fixed), multi=False
                )
        else:
            n_sources = len(neighbour_catalog["x"])
            # PySersicMultiPrior (unlike SourceProperties) does not
            # estimate the sky background itself -- it requires an
            # explicit guess whenever sky_type != "none". Reuse
            # SourceProperties' own edge-pixel sky estimate for
            # consistency with the single-source path.
            sky_props = SourceProperties(sci, mask=mask)
            prior = PySersicMultiPrior(
                neighbour_catalog,
                sky_type="flat",
                sky_guess=sky_props.sky_guess,
                sky_guess_err=sky_props.sky_guess_err,
            )
            self._apply_prior_overrides(
                prior, self.primary_fixed, self.primary_bounded, suffix="_0"
            )
            for i in range(1, n_sources):
                self._apply_prior_overrides(
                    prior,
                    self.neighbour_fixed,
                    self.neighbour_bounded,
                    suffix=f"_{i}",
                )
            for i in range(n_sources):
                if f"xc_{i}" in prior.dist_dict:
                    prior.set_gaussian_prior(
                        f"xc_{i}",
                        float(neighbour_catalog["x"][i]),
                        self.xy_center_sigma,
                    )
                    prior.set_gaussian_prior(
                        f"yc_{i}",
                        float(neighbour_catalog["y"][i]),
                        self.xy_center_sigma,
                    )
            fitter = FitMulti(
                data=sci,
                rms=rms,
                mask=mask,
                psf=psf_arr,
                prior=prior,
                renderer_kwargs={"use_interp_amps": True},
            )
            fixed_keys = {f"{p}_0": v for p, v in self.primary_fixed.items()}
            for i in range(1, n_sources):
                fixed_keys.update(
                    {f"{p}_{i}": v for p, v in self.neighbour_fixed.items()}
                )
            if fixed_keys:
                self._patch_fixed_params_build_model(
                    fitter, fixed_keys, multi=True
                )

        fitter.estimate_posterior(
            rkey=jax.random.PRNGKey(self.rkey_seed), method=self.method
        )
        model_image = np.asarray(
            fitter.svi_results.models.median(dim=["chain", "draw"]).values
        )
        map_params = fitter.svi_results.retrieve_param_quantiles(
            return_dataframe=False
        )
        map_params["model"] = model_image
        return map_params, fitter.svi_results

    def _fit_cutout(
        self: Self,
        cutout: Type[Band_Cutout_Base],
        cat: Optional[Catalogue] = None,
        plot: bool = True,
        overwrite: bool = False,
        out_dir: str = "",
        *args: Any,
        **kwargs: Dict[str, Any],
    ) -> Optional[PySersic_Result]:
        sci = cutout.load("SCI")[1]
        rms = cutout.load("RMS_ERR")[1]
        seg_header, seg_map = cutout.load("SEG")

        mask = np.zeros_like(seg_map, dtype=bool)
        center_y, center_x = np.array(seg_map.shape) // 2
        source_id = seg_map[center_y, center_x]
        mask[np.where(seg_map != source_id)] = True
        mask[np.where(seg_map == 0)] = False

        psf_full = self.psf.cutout.band_data.load_im()[0]
        psf_center = (psf_full.shape[1] / 2.0, psf_full.shape[0] / 2.0)
        psf_arr = Cutout2D(
            psf_full,
            position=psf_center,
            size=sci.shape,
            mode="partial",
            fill_value=0.0,
        ).data
        psf_arr = (psf_arr / np.sum(psf_arr)).astype(np.float32)

        sci = sci.astype(np.float32)
        rms = rms.astype(np.float32)

        invalid_rms = ~np.isfinite(rms) | (rms <= 0) | ~np.isfinite(sci)
        if np.any(invalid_rms):
            mask[invalid_rms] = True
            rms[invalid_rms] = (
                np.nanmax(rms[~invalid_rms]) if np.any(~invalid_rms) else 1.0
            )

        neighbour_catalog = None
        n_neighbours = 0
        if self.neighbours_model is not None and cat is not None:
            neighbours = self._get_neighbours(cutout, out_dir, cat=cat)
            n_neighbours = len(neighbours["ID"])
            if n_neighbours > 0:
                pix_scale = cutout.band_data.pix_scale.to(u.arcsec).value
                zp = cutout.band_data.ZP
                flux = 10 ** ((zp - np.asarray(neighbours["MAG_AUTO"])) / 2.5)
                r_pix = np.asarray(neighbours["FLUX_RADIUS"]) / pix_scale
                theta_rad = np.deg2rad(np.asarray(neighbours["THETA_IMAGE"]))
                neighbour_catalog = {
                    "x": np.concatenate(
                        [[float(center_x)], neighbours["X_IMAGE"]]
                    ),
                    "y": np.concatenate(
                        [[float(center_y)], neighbours["Y_IMAGE"]]
                    ),
                    "flux": np.concatenate(
                        [[float(np.nansum(sci[~mask]))], flux]
                    ),
                    "r": np.concatenate([[3.0], r_pix]),
                    "theta": np.concatenate([[0.0], theta_rad]),
                    "type": [self.model]
                    + [self.neighbours_model] * n_neighbours,
                }

        label = f"{cutout.ID}"
        model_dir = self.model + ("_with_neighbours" if n_neighbours else "")
        for name, params in [
            ("fixed", self.primary_fixed),
            ("bounded", self.primary_bounded),
        ]:
            for param, val in params.items():
                model_dir += (
                    f"_{name}_{param}={val:.2f}"
                    if name == "fixed"
                    else f"_{name}_{param}={val[0]:.2f}-{val[1]:.2f}"
                )
        out_path = f"{out_dir}/{model_dir}/{label}_pysersic.h5"
        residual_path = f"{out_dir}/{model_dir}/{label}_residual.png"

        if Path(out_path).is_file() and not overwrite:
            with h5py.File(out_path, "r") as hf:
                map_params = {
                    key: np.asarray(hf["map_params"][key])
                    for key in hf["map_params"].keys()
                }
                samples_by_key = {
                    key: np.asarray(hf["samples"][key])
                    for key in hf["samples"].keys()
                }
        else:
            map_params, svi_results = self._fit_array(
                sci,
                rms,
                mask,
                psf_arr,
                neighbour_catalog=neighbour_catalog,
            )
            samples_by_key = {
                key: np.asarray(
                    svi_results.idata.posterior[key].values
                ).flatten()
                for key in map_params.keys()
                if key != "model"
            }
            funcs.make_dirs(out_path)
            with h5py.File(out_path, "w") as hf:
                hf.create_dataset("sci", data=sci)
                hf.create_dataset("rms", data=rms)
                hf.create_dataset("mask", data=mask)
                hf.create_dataset("psf", data=psf_arr)
                map_params_grp = hf.create_group("map_params")
                for key, val in map_params.items():
                    map_params_grp.create_dataset(key, data=np.asarray(val))
                samples_grp = hf.create_group("samples")
                for key, val in samples_by_key.items():
                    samples_grp.create_dataset(key, data=val)

        primary_suffix = "_0" if neighbour_catalog is not None else ""
        # every key trivially ends with "" when primary_suffix=="", so
        # this naturally covers both the single- and multi-source cases
        base_profile_params_keys = [
            k
            for k in map_params.keys()
            if k != "model" and k.endswith(primary_suffix)
        ]
        properties: Dict[str, u.Quantity] = {}
        property_errs: Dict[str, List[u.Quantity]] = {}
        property_pdfs: Dict[str, PDF] = {}
        param_samples: Dict[str, np.ndarray] = {}
        for key in base_profile_params_keys:
            name = (
                key[: -len(primary_suffix)]
                if primary_suffix and key.endswith(primary_suffix)
                else key
            )
            quantiles = np.asarray(map_params[key])
            unit = u.dimensionless_unscaled
            properties[name] = float(quantiles[1]) * unit
            property_errs[name] = [
                float(quantiles[1] - quantiles[0]) * unit,
                float(quantiles[2] - quantiles[1]) * unit,
            ]
            property_pdfs[name] = PDF.from_1D_arr(
                name, samples_by_key[key] * unit
            )
            param_samples[name] = samples_by_key[key]

        neighbour_properties = []
        if neighbour_catalog is not None:
            for i in range(1, len(neighbour_catalog["x"])):
                suffix = f"_{i}"
                neighbour_properties.append(
                    {
                        key[: -len(suffix)]: float(
                            np.asarray(map_params[key])[1]
                        )
                        for key in map_params.keys()
                        if key != "model" and key.endswith(suffix)
                    }
                )

        chi2 = float(np.sum(((sci - map_params["model"]) / rms)[~mask] ** 2))
        Ndof = int(np.sum(~mask)) - len(base_profile_params_keys)

        result = PySersic_Result(
            fitter=self,
            chi2=chi2,
            Ndof=Ndof,
            properties=properties,
            property_errs=property_errs,
            property_pdfs=property_pdfs,
            model_image=map_params["model"],
            neighbour_properties=neighbour_properties,
            param_samples=param_samples,
            cutout=cutout,
        )
        result._fit_mask = mask

        if plot:
            result.plot(save=True, out_path=residual_path)

        cutout.update_morph_fits(result)
        return result

    def _fit_cat(
        self: Self,
        cat: Catalogue,
        fit_filt: Filter,
        native: bool = False,
        plot: bool = True,
        out_dir: str = "",
        *args: Any,
        **kwargs: Dict[str, Any],
    ) -> Table:
        cat.make_band_cutouts(
            fit_filt,
            self.psf.cutout.cutout_size,
            native=native,
        )
        cutout_key = (
            f"{fit_filt.filt_name}_"
            f"{self.psf.cutout.cutout_size.to(u.arcsec).value:.2f}as"
        )
        if native:
            cutout_key += "_native"

        if out_dir == "":
            # mirrors Galfit_Fitter._fit_cat's default output-directory
            # convention, under the [PYSERSIC] section of the galfind
            # config rather than [GALFIT]
            out_dir = (
                f"{config['PYSERSIC']['OUTPUT_DIR']}/{cat.version}"
                + f"/{cat.filterset.instrument_name}/"
                + funcs.aper_diams_to_str(cat.aper_diams)
                .replace("(", "")
                .replace(")", "")
                + f"/{cat.survey}/{fit_filt.filt_name}"
            )
            if native:
                out_dir += "_native"

        results = [
            self._fit_cutout(
                gal.cutouts[cutout_key],
                cat=cat,
                plot=plot,
                out_dir=f"{out_dir}/{gal.ID}",
                *args,
                **kwargs,
            )
            for gal in tqdm(
                cat,
                total=len(cat),
                desc=f"fitting/loading {repr(self)} for {repr(cat)}",
            )
        ]
        out_path = f"{out_dir}/{self.model}/results.fits"
        if not Path(out_path).is_file():
            tab = self._make_results_table(results, out_path=out_path)
        else:
            tab = Table.read(out_path)
        return tab
