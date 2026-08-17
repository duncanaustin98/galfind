"""MCMC prior distributions and posterior sampling utilities.

Provides Prior abstract base class with implementations (Flat, Gaussian) for
MCMC sampling and parameter estimation with posterior visualization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import emcee
import numpy as np
import corner
import multiprocessing as mp
import json
from astropy.stats import sigma_clip
from tqdm import tqdm
import h5py
from matplotlib import patheffects as pe
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import NoReturn, Union, Optional, List, Dict, Any, Tuple
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from .. import galfind_logger
from ..utils import useful_funcs_austind as funcs

class Prior(ABC):
    """Abstract base class for a single-parameter MCMC prior distribution.

    Subclasses implement `get_initpos` to draw a random starting position
    for an MCMC walker and `__call__` to evaluate the (unnormalised)
    log-prior probability of a parameter value.

    Parameters
    ----------
    name : `str`
        Name of the parameter this prior applies to.
    prior_params : `dict` of `str`: `float`
        Parameters defining the shape of the prior distribution (e.g.
        limits, mean, standard deviation).
    fiducial : `float`
        Fiducial/reference value of the parameter, used e.g. for plotting
        or as a default initial position.

    Attributes
    ----------
    name : `str`
        Name of the parameter this prior applies to.
    prior_params : `dict` of `str`: `float`
        Parameters defining the shape of the prior distribution.
    fiducial : `float`
        Fiducial/reference value of the parameter.
    """

    def __init__(
        self: Self,
        name: str,
        prior_params: Dict[str, float],
        fiducial: float
    ):
        self.name = name
        self.prior_params = prior_params
        self.fiducial = fiducial

    # @abstractmethod
    # @property
    # def latex_name(self: Self) -> str:
    #     """Return the LaTeX name of the prior."""
    #     pass

    @abstractmethod
    def get_initpos(
        self: Self,
    ) -> float:
        """Return an initial position for the MCMC walkers."""
        pass
    
    @abstractmethod
    def __call__(
        self: Self,
        param: float
    ) -> bool:
        pass

    def __add__(
        self: Self,
        other: Union[Type[Self], Type[Priors]]
    ) -> Self:
        if isinstance(other, tuple(Prior.__subclasses__())):
            assert self.name != other.name, \
                galfind_logger.critical(
                    "Prior names must be unique"
                )
            prior_arr = [self, other]
        else:
            assert not self.name in other.names, \
                galfind_logger.critical(
                    "Prior names must be unique"
                )
            prior_arr = [self] + other.prior_arr
        return Priors(prior_arr)
    

class Flat_Prior(Prior):
    """`Prior` subclass implementing a flat (uniform) prior distribution.

    The log-prior probability is 0 within the given limits and
    ``-numpy.inf`` (i.e. disallowed) outside them.

    Parameters
    ----------
    name : `str`
        Name of the parameter this prior applies to.
    prior_lims : `list` of `float`
        Two-element list ``[lower_lim, upper_lim]`` giving the (ascending)
        limits of the flat prior.
    fiducial : `float`
        Fiducial/reference value of the parameter.

    Attributes
    ----------
    name : `str`
        Name of the parameter this prior applies to.
    prior_params : `dict`
        Dictionary with keys ``"lower_lim"`` and ``"upper_lim"`` giving the
        limits of the flat prior.
    fiducial : `float`
        Fiducial/reference value of the parameter.
    """

    def __init__(
        self: Self,
        name: str,
        prior_lims: List[float],
        fiducial: float,
    ):
        assert len(prior_lims) == 2, \
            galfind_logger.critical(
                "Flat priors must have two limits"
            )
        assert prior_lims[0] < prior_lims[1], \
            galfind_logger.critical(
                "Flat prior limits must be in ascending order"
            )
        prior_params = {"lower_lim": prior_lims[0], "upper_lim": prior_lims[1]}
        super().__init__(name, prior_params, fiducial)

    def get_initpos(
        self: Self,
    ) -> float:
        """Draw a random initial walker position from the flat prior.

        Returns
        -------
        `float`
            A value drawn uniformly from ``[lower_lim, upper_lim]``.
        """
        return np.random.uniform(self.prior_params["lower_lim"], self.prior_params["upper_lim"], 1)

    def __call__(
        self: Self,
        param: float
    ) -> float:
        if param > self.prior_params["lower_lim"] and \
                param < self.prior_params["upper_lim"]:
            return 0.0 #-np.log(self.prior_params["upper_lim"] - self.prior_params["lower_lim"])
        else:
            return -np.inf


class Gaussian_Prior(Prior):
    """`Prior` subclass implementing a Gaussian (normal) prior distribution.

    Parameters
    ----------
    name : `str`
        Name of the parameter this prior applies to.
    prior_lims : `dict` of `str`: `float`
        Dictionary with keys ``"mu"`` and ``"sigma"`` giving the mean and
        standard deviation of the Gaussian prior.
    fiducial : `float`
        Fiducial/reference value of the parameter.

    Attributes
    ----------
    name : `str`
        Name of the parameter this prior applies to.
    prior_params : `dict`
        Dictionary with keys ``"mu"`` and ``"sigma"`` giving the mean and
        standard deviation of the Gaussian prior.
    fiducial : `float`
        Fiducial/reference value of the parameter.
    """

    def __init__(
        self: Self,
        name: str,
        prior_lims: Dict[str, float],
        fiducial: float,
    ):
        assert len(prior_lims) == 2, \
            galfind_logger.critical(
                "Flat priors must have two limits"
            )
        assert all(key in prior_lims.keys() for key in ["mu", "sigma"]), \
            galfind_logger.critical(
                f"{prior_lims.keys()=} does not contain 'mu' and 'sigma' keys!"
            )
        prior_params = prior_lims
        super().__init__(name, prior_params, fiducial)

    def get_initpos(
        self: Self,
    ) -> float:
        """Draw a random initial walker position from the Gaussian prior.

        Returns
        -------
        `float`
            A value drawn from a normal distribution with mean ``mu`` and
            standard deviation ``sigma``.
        """
        return np.random.normal(self.prior_params["mu"], self.prior_params["sigma"], 1)

    def __call__(
        self: Self,
        param: float
    ) -> float:
        return -0.5 * ((param - self.prior_params["mu"]) / self.prior_params["sigma"]) ** 2
           # - np.log(self.prior_params["sigma"]) - 0.5 * np.log(2 * np.pi)


class Priors:
    """A collection of `Prior` objects for a multi-parameter MCMC fit.

    Supports addition (with another `Priors` instance or a single `Prior`),
    indexing by parameter name, iteration, and `len`.

    Parameters
    ----------
    prior_arr : `list` of `Prior`
        List of individual `Prior` objects making up this collection.

    Attributes
    ----------
    prior_arr : `list` of `Prior`
        List of individual `Prior` objects making up this collection.
    """

    def __init__(
        self: Self,
        prior_arr: List[Prior],
    ):
        self.prior_arr = prior_arr

    @property
    def names(self):
        """`list` of `str`: Names of every `Prior` in this collection."""
        return [prior.name for prior in self.prior_arr]

    def __call__(
        self: Self,
        params: Dict[str, float]
    ) -> float:
        return np.sum(prior(params[prior.name]) for prior in self.prior_arr)

    def __add__(
        self: Self,
        other: Union[Self, Type[Prior]]
    ) -> Self:
        if isinstance(other, tuple(Prior.__subclasses__())):
            assert other.name not in self.names, \
                galfind_logger.critical(
                    "Prior names must be unique"
                )
            self.prior_arr = self.prior_arr + [other]
        else:
            self.prior_arr = self.prior_arr + other.prior_arr
        return self
    
    def __getitem__(
        self: Self,
        name: str
    ) -> Prior:
        for prior in self.prior_arr:
            if prior.name == name:
                return prior
        galfind_logger.critical(
            f"{name=} not found in {self.names=}"
        )
    
    def __len__(
        self: Self
    ) -> int:
        return len(self.prior_arr)
    
    def __iter__(
        self: Self
    ) -> Priors:
        self._iter_index = 0
        return self
    
    def __next__(
        self: Self
    ) -> Prior:
        if self._iter_index < len(self.prior_arr):
            self._iter_index += 1
            return self.prior_arr[self._iter_index - 1]
        else:
            raise StopIteration
    
    def __repr__(
        self: Self
    ) -> str:
        return f"Priors({self.names})"
    

class Base_MCMC_Fitter(ABC):
    """Abstract base class wrapping an `emcee.EnsembleSampler` MCMC fit.

    Handles walker initialisation, running the sampler (optionally with an
    HDF5 backend for persistence/resuming), and generic post-processing
    such as flattening/thinning chains, computing BIC/AIC, and plotting the
    resulting fit and corner plots. Subclasses must implement `update`,
    `model` and `log_likelihood` to define the specific model being fit.

    Parameters
    ----------
    priors : `Priors`
        Collection of priors, one per free parameter, to fit for.
    x_data : `NDArray` of `float`
        Independent variable data values.
    y_data : `NDArray` of `float`
        Dependent variable data values.
    y_data_errs : `NDArray` of `NDArray` of `float`
        Lower/upper (or symmetric) errors on `y_data`.
    nwalkers : `int`
        Number of MCMC walkers to use in the `emcee.EnsembleSampler`.
    backend_filename : `str` or `None`
        Path to an HDF5 file used as an `emcee.backends.HDFBackend` for
        storing/resuming the chain. If `None`, no backend is used and the
        chain is kept in memory only.
    fixed_params : `dict` of `str`: `float`
        Parameters that are held fixed (not fit for) at the given values.
    init_pos : `NDArray` of `float`, optional
        Initial walker positions. If `None`, positions are drawn from the
        priors via `_instantiate_walkers`. Default is `None`.

    Attributes
    ----------
    priors : `Priors`
        Collection of priors, one per free parameter, being fit for.
    x_data : `NDArray` of `float`
        Independent variable data values.
    y_data : `NDArray` of `float`
        Dependent variable data values.
    y_data_errs : `NDArray` of `NDArray` of `float`
        Lower/upper (or symmetric) errors on `y_data`.
    nwalkers : `int`
        Number of MCMC walkers used in the `emcee.EnsembleSampler`.
    backend : `emcee.backends.HDFBackend`
        HDF5-backed sampler backend, only set if `backend_filename` is not
        `None`.
    sampler : `emcee.EnsembleSampler`
        The underlying `emcee` ensemble sampler.
    backend_filename : `str` or `None`
        Path to the HDF5 backend file, if any.
    init_pos : `NDArray` of `float`
        Initial walker positions used to start/resume the sampler.
    fixed_params : `dict` of `str`: `float`
        Parameters that are held fixed (not fit for) at the given values.
    """

    def __init__(
        self: Self,
        priors: Priors,
        x_data: NDArray[float],
        y_data: NDArray[float],
        y_data_errs: NDArray[NDArray[float, float]],
        nwalkers: int,
        backend_filename: Optional[str],
        fixed_params: Dict[str, float],
        init_pos: Optional[NDArray[float]] = None,
    ):
        self.priors = priors
        self.x_data = x_data
        self.y_data = y_data
        self.y_data_errs = y_data_errs
        self.nwalkers = nwalkers
        if backend_filename is not None:
            if backend_filename.split(".")[-1] != "h5":
                backend_filename += ".h5"
            funcs.make_dirs(backend_filename)
            self.backend = emcee.backends.HDFBackend(backend_filename)
            try:
                self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_likelihood, backend = self.backend) #, blobs_dtype = blobs_dtype, pool = pool)
            except Exception as e:
                with h5py.File(backend_filename, "r") as f:
                    print(list(f.keys()))
                    for k in f["mcmc"]:
                        print(k, f["mcmc"][k].shape)
                err_message = f"{e}: Could not load {backend_filename=}! Delete the file or choose a different name."
                galfind_logger.critical(err_message)
                raise Exception(err_message)
        else:
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_likelihood)
        self.backend_filename = backend_filename
        if init_pos is not None:
            self.init_pos = init_pos
        else:
            self._instantiate_walkers()
        self.fixed_params = fixed_params

    # TODO:
    # @abstractmethod
    # @classmethod
    # def from_h5(cls: Type[Self], h5_path: str) -> Self:
    #     """Load MCMC fitter from an HDF5 file."""
    #     pass

    @property
    def ndim(self):
        """`int`: Number of free (fitted) parameters, i.e. the number of priors."""
        return len(self.priors)

    @property
    def fiducial_params(self):
        """`list` of `float`: Fiducial value of every prior, in prior order."""
        return [prior.fiducial for prior in self.priors]

    def get_BIC(self):
        """Compute the Bayesian Information Criterion (BIC) of the fit.

        Uses the maximum log-likelihood found so far in the sampler
        `backend`, the number of free parameters, and the number of data
        points.

        Returns
        -------
        `float` or `None`
            The BIC value, or `None` if no `backend` exists or the backend
            has not yet accumulated any iterations.
        """
        if not hasattr(self, "backend"):
            BIC = None
        backend_len = self.backend.iteration
        if backend_len == 0:
            BIC = None
        else:
            lnL = self.backend.get_log_prob(flat = True)
            max_lnL = np.max(lnL)
            k = len(self.priors) # free params length
            n = len(self.x_data) # data length
            BIC = k * np.log(n) - 2 * max_lnL
        galfind_logger.info(f"BIC: {BIC}")
        return BIC

    def get_AIC(self):
        """Compute the Akaike Information Criterion (AIC) of the fit.

        Uses the maximum log-likelihood found so far in the sampler
        `backend` and the number of free parameters.

        Returns
        -------
        `float` or `None`
            The AIC value, or `None` if no `backend` exists or the backend
            has not yet accumulated any iterations.
        """
        if not hasattr(self, "backend"):
            AIC = None
        backend_len = self.backend.iteration
        if backend_len == 0:
            AIC = None
        else:
            lnL = self.backend.get_log_prob(flat = True)
            max_lnL = np.max(lnL)
            k = len(self.priors) # free params length
            n = len(self.x_data) # data length
            AIC = 2 * (k - max_lnL)
        galfind_logger.info(f"AIC: {AIC}")
        return AIC
    
    def get_autocorr_time(self) -> int:
        """Estimate the integrated autocorrelation time of the chains.

        Falls back to ``backend.iteration / 50`` if `emcee` is unable to
        reliably estimate the autocorrelation time (e.g. because the
        chains are not yet converged).

        Returns
        -------
        `int`
            The (maximum, over parameters) estimated autocorrelation time,
            used elsewhere to set chain discard/thin values.
        """
        try:
            sampler_autocorr_time = self.sampler.get_autocorr_time()
            autocorr_time = np.nanmax(sampler_autocorr_time)
        except Exception as e:
            galfind_logger.warning(
                f"Could not calculate autocorrelation time! Chains likely unconverged! {e}"
            )
            autocorr_time = self.backend.iteration / 50
        return autocorr_time
    
    def _instantiate_walkers(self: Self) -> NoReturn:
        # # uniformly distributed starting positions over flat prior
        # flat_priors_lower = [self.flat_priors[i][0] for i in range(self.ndim)]
        # flat_priors_diff = [self.flat_priors[i][1] - self.flat_priors[i][0] for i in range(self.ndim)]

        # pos = [flat_priors_lower + (flat_priors_diff) \
        #        * np.random.uniform(0, 1, self.ndim) for i in range(self.nwalkers)]
        if not hasattr(self, "init_pos"):
            # init_pos = [self.fiducial_params + 1e-4 * np.random.uniform(0, 1, self.ndim) * \
            #         self.fiducial_params for i in range(self.nwalkers)]
            init_pos = [np.array([prior.get_initpos()[0] for prior in self.priors]) \
                for i in range(self.nwalkers)]
            self.init_pos = init_pos

    def __call__(
        self: Self,
        n_steps: int,
        n_processes: int = 1,
    ) -> None:
        if hasattr(self, "backend"):
            galfind_logger.info("Initial size: {0}".format(self.backend.iteration))
            if n_steps <= self.backend.iteration:
                galfind_logger.warning(
                    f"n_steps={n_steps=}<={self.backend.iteration=}"
                )
                return
            else:
                n_steps -= self.backend.iteration
        else:
            galfind_logger.info("Initial size: 0")
        with mp.Pool(processes=n_processes) as pool:
            # blobs_dtype = [(key, object) for key in self.blob_keys]
            # print("blobs_dtype = ", blobs_dtype)
            self.sampler.run_mcmc(self.init_pos, n_steps, progress = True)
            if hasattr(self, "backend"):
                galfind_logger.info("Final size: {0}".format(self.backend.iteration))
        pool.close()

    @abstractmethod
    def update(self):
        """Update any derived/cached fitter state (e.g. after loading a chain).

        Abstract method to be implemented by subclasses.
        """
        pass

    @abstractmethod
    def model(
        self: Self,
        x: Union[float, List[float], NDArray[float]],
        params: Dict[str, float],
        **kwargs: Dict[str, Any],
    ) -> float:
        """Evaluate the model at ``x`` for the given parameters.

        Abstract method to be implemented by subclasses.

        Parameters
        ----------
        x : `float`, `list` of `float`, or `NDArray` of `float`
            Independent variable value(s) at which to evaluate the model.
        params : `dict` of `str`: `float`
            Model parameter values, keyed by parameter name.
        **kwargs : `dict`
            Additional model-specific keyword arguments.

        Returns
        -------
        `float`
            The model value(s) evaluated at ``x``.
        """
        pass

    # @abstractmethod
    # def blob_funcs(self):
    #     pass

    @abstractmethod
    def log_likelihood(
        self: Self,
        params: List[float],
    ) -> float:
        """Compute the log-likelihood of the data given the parameters.

        Abstract method to be implemented by subclasses.

        Parameters
        ----------
        params : `list` of `float`
            Parameter values, in the same order as `priors`.

        Returns
        -------
        `float`
            The log-likelihood of the data given ``params``.
        """
        pass

    def _fix_params(
        self: Self,
        params: Dict[str, float]
    ) -> Dict[str, float]:
        for key, val in self.fixed_params.items():
            params[key] = val
        return params
    
    # def _calculate_scatter(self) -> float:
    #     residuals = self.get_residuals(self.get_params_med())
    #     self.scatter = np.sqrt(np.sum(residuals ** 2) / (len(self.y_data) - 1))
    #     galfind_logger.info(f"Scatter: {self.scatter:.3f} dex")
    #     return self.scatter

    # def get_sample(self: Self) -> NDArray[float]:
    #     autocorr_time = np.max(self.sampler.get_autocorr_time())
    #     discard = int(autocorr_time * 2)
    #     thin = int(autocorr_time / 2)
    #     chain = self.sampler.get_chain(flat = True, discard = discard, thin = thin)
    #     return chain

    # def get_params_med(self: Self) -> Dict[str, float]:
    #     if not hasattr(self, "params_med"):
    #         chain = self.get_sample()
    #         self.params_med = {prior.name: np.median(chain[:, i]) for i, prior in enumerate(self.priors)}
    #     galfind_logger.info(f"Median parameters: {self.params_med}")
    #     return self.params_med

    def plot(
        self: Self,
        ax: plt.Axes,
        log_data: bool = False,
        x_arr: Optional[NDArray[float]] = None,
        plot_kwargs: Dict[str, Any] = {},
        fill_between_kwargs: Dict[str, Any] = {},
        xoff: float = 0.0,
        **kwargs: Dict[str, Any],
    ) -> None: #nsamples = 1_000, plot_med = False):
        """Plot the median MCMC model fit and its 1-sigma band on an axis.

        Parameters
        ----------
        ax : `plt.Axes`
            Matplotlib axis to plot the fit on.
        log_data : `bool`, optional
            Whether to plot `model` evaluated in log10 space. Default is
            `False`.
        x_arr : `NDArray` of `float`, optional
            x-values at which to evaluate the model. If `None`, 100
            linearly spaced values spanning `x_data` are used. Default is
            `None`.
        plot_kwargs : `dict`, optional
            Keyword arguments passed to `ax.plot` for the median line,
            overriding the defaults. Default is `{}`.
        fill_between_kwargs : `dict`, optional
            Keyword arguments passed to `ax.fill_between` for the 1-sigma
            band, overriding the defaults. Default is `{}`.
        xoff : `float`, optional
            Offset subtracted from `x_arr` before plotting. Default is
            ``0.0``.
        **kwargs : `dict`
            Additional keyword arguments passed on to `model` via
            `_get_plot_chains`.

        Returns
        -------
        `tuple`
            ``(x_arr, med_chains, l1_chains, u1_chains)``: the x-values
            plotted and the corresponding median, 16th and 84th percentile
            model chains.
        """

        def_plot_kwargs = {
            "color": "black",
            "ls": "--",
            "zorder": 200,
            "path_effects": [pe.withStroke(linewidth = 2., foreground = "white")],
        }
        for key, value in plot_kwargs.items():
            def_plot_kwargs[key] = value
        def_fill_between_kwargs = {
            "color": def_plot_kwargs["color"],
            "alpha": 0.5,
            "zorder": 199,
            "path_effects": [pe.withStroke(linewidth = 2., foreground = "white")],
        }
        for key, value in fill_between_kwargs.items():
            def_fill_between_kwargs[key] = value

        if x_arr is None:
            x_arr = np.linspace(np.min(self.x_data), np.max(self.x_data), 100)
        l1_chains, med_chains, u1_chains = self._get_plot_chains(x_arr = x_arr, log_data = log_data, **kwargs)
        ax.plot(x_arr - xoff, med_chains, **def_plot_kwargs)
        ax.fill_between(x_arr - xoff, l1_chains, u1_chains, **def_fill_between_kwargs)
        galfind_logger.info("Plotting MCMC fit")
        return x_arr, med_chains, l1_chains, u1_chains

    def get_sample(self: Self, discard: bool = True, thin: bool = True) -> NDArray[float]:
        """Return a flattened sample of the MCMC chain.

        Parameters
        ----------
        discard : `bool`, optional
            Whether to discard the first ``2 * autocorr_time`` steps as
            burn-in. Default is `True`.
        thin : `bool`, optional
            Whether to thin the chain by ``autocorr_time / 2``. Default is
            `True`.

        Returns
        -------
        `NDArray` of `float`
            The flattened (and optionally discarded/thinned) parameter
            chain, of shape ``(n_samples, ndim)``.
        """
        autocorr_time = self.get_autocorr_time()
        if discard:
            discard_ = int(autocorr_time * 2)
        else:
            discard_ = 0
        if thin:
            thin_ = int(autocorr_time / 2)
        else:
            thin_ = 1
        chain = self.sampler.get_chain(flat = True, discard = discard_, thin = thin_)
        return chain
    
    def _get_plot_chains(
        self: Self,
        x_arr: NDArray[float],
        log_data: bool = False,
        **kwargs: Dict[str, Any],
    ) -> Tuple[NDArray[float], NDArray[float], NDArray[float]]:
        y_fit = self.get_chains(x_arr, log_data = log_data, **kwargs).T
        l1_chains = np.array([np.percentile(y, 16) for y in y_fit])
        med_chains = np.array([np.percentile(y, 50) for y in y_fit])
        u1_chains = np.array([np.percentile(y, 84) for y in y_fit])
        return l1_chains, med_chains, u1_chains
    
    def get_chains(
        self: Self,
        x_arr: NDArray[float],
        log_data: bool = False,
        shape: Optional[int] = None,
        incl_scatter: bool = False,
        **kwargs: Dict[str, Any],
    ) -> NDArray[float]:
        """Evaluate `model` over (a subset of) the posterior chain at ``x_arr``.

        Parameters
        ----------
        x_arr : `NDArray` of `float`
            x-values at which to evaluate the model for every chain sample.
        log_data : `bool`, optional
            Whether to evaluate `model` in log10 space. Default is `False`.
        shape : `int`, optional
            If given, only the last ``shape`` chain samples are used
            instead of the full (discarded/thinned) chain. Default is
            `None`.
        incl_scatter : `bool`, optional
            Whether to perturb the ``"c"`` parameter of each sample by a
            random draw from `get_scatter` before evaluating the model.
            Default is `False`.
        **kwargs : `dict`
            Additional keyword arguments passed on to `model`.

        Returns
        -------
        `NDArray` of `float`
            Array of model values evaluated at `x_arr` for each chain
            sample, of shape ``(n_samples, len(x_arr))``.
        """
        autocorr_time = self.get_autocorr_time()
        discard = int(autocorr_time * 2)
        thin = int(autocorr_time / 2)

        if log_data:
            model = lambda x, params: np.log10(self.model(x, params, **kwargs))
        else:
            model = self.model
        if shape is None:
            params_arr = self.sampler.get_chain(flat = True, discard = discard, thin = thin)
        else:
            params_arr = self.sampler.get_chain(flat = True, discard = discard, thin = thin)[-shape:]
        if incl_scatter:
            scatter = self.get_scatter()
            params_dict_arr = [
                {
                    **{
                        prior.name: param if not prior.name == "c" else np.random.normal(param, scatter[0])
                        if param < self.get_params_med()[prior.name] else np.random.normal(param, scatter[1])
                        for prior, param in zip(self.priors, params)
                    },
                    **self.fixed_params
                }
                for params in params_arr
            ]
        else:
            params_dict_arr = [
                {
                    **{prior.name: param for prior, param in zip(self.priors, params)},
                    **self.fixed_params
                }
                for params in params_arr
            ]
        y_fit = np.array([model(x_arr, params_dict, **kwargs) for params_dict in tqdm( \
            params_dict_arr, desc = "Loading chains", total = len(params_dict_arr))])
        return y_fit

    def plot_corner(
        self: Self,
        fig: Optional[plt.Figure] = None,
        range: Optional[List[Tuple[float]]] = None,
        colour: str = "black",
        legend: bool = False,
        save: bool = True,
        fid_colour: str = "C1",
        discard: Optional[Union[int, str]] = "default",
        thin: Optional[Union[int, str]] = "default",
        **plot_kwargs: Dict[str, Any]
    ) -> plt.Figure:
        """Make a `corner` plot of the posterior parameter distributions.

        Parameters
        ----------
        fig : `plt.Figure`, optional
            Existing figure to draw the corner plot onto. If `None`, a new
            figure is created. Default is `None`.
        range : `list` of `tuple` of `float`, optional
            Per-parameter axis ranges, passed to `corner.corner`. Must have
            the same length as the number of parameter labels if given.
            Default is `None`.
        colour : `str`, optional
            Colour of the posterior contours/histograms. Default is
            ``"black"``.
        legend : `bool`, optional
            Whether to add a legend to the figure. Default is `False`.
        save : `bool`, optional
            Whether to save the resulting figure to
            ``backend_filename`` with ``.h5`` replaced by ``_MCMC.jpeg``.
            Default is `True`.
        fid_colour : `str`, optional
            Colour used to overplot the `fiducial_params` values on the
            corner plot. If `None`, fiducial values are not overplotted.
            Default is ``"C1"``.
        discard : `int`, `str`, or `None`, optional
            Number of burn-in steps to discard from the chain. If
            ``"default"``, uses ``2 * autocorr_time``. If `None`, no steps
            are discarded. Default is ``"default"``.
        thin : `int`, `str`, or `None`, optional
            Chain thinning factor. If ``"default"``, uses
            ``autocorr_time / 2``. If `None`, the chain is not thinned.
            Default is ``"default"``.
        **plot_kwargs : `dict`
            Additional keyword arguments passed to `corner.corner`,
            overriding the defaults.

        Returns
        -------
        `plt.Figure`
            The `corner` plot figure.
        """

        autocorr_time = self.get_autocorr_time()
        get_chain_kwargs = {}
        if discard == "default":
            discard = int(autocorr_time * 2)
        if discard is not None:
            get_chain_kwargs["discard"] = discard
        if thin == "default":
            thin = int(autocorr_time / 2)
        if thin is not None:
            get_chain_kwargs["thin"] = thin
        flat_samples = self.backend.get_chain(flat = True, **get_chain_kwargs) #discard = discard, thin = thin)

        if "labels" not in plot_kwargs.keys():
            plot_kwargs["labels"] = [prior.name for prior in self.priors]
        if range is not None:
            assert len(range) == len(plot_kwargs["labels"]), \
                galfind_logger.critical(
                    f"{len(range)=} must be equal to {len(plot_kwargs['labels'])=}"
                )

        if fig is None:
            fig_corner = plt.figure()
        else:
            fig_corner = fig
        default_plot_kwargs = {
            "fig": fig_corner,
            "color": colour,
            "quantiles": [0.16, 0.5, 0.84],
            "smooth": None,
            "title_kwargs": {"fontsize": 12},
            "show_titles": True,
            #"sigma_arr": [1, 2, 3],
            "plot_density": False,
            "plot_datapoints": False,
        }
        for key, value in plot_kwargs.items():
            default_plot_kwargs[key] = value
        # convert sigma_arr to levels
        if "sigma_arr" in default_plot_kwargs.keys():
            if not "levels" in default_plot_kwargs.keys():
                default_plot_kwargs["levels"] = tuple([ \
                    1.0 - np.exp(-0.5 * np.array(sigma) ** 2) \
                    for sigma in default_plot_kwargs["sigma_arr"]])
            del default_plot_kwargs["sigma_arr"]
        if len(default_plot_kwargs["quantiles"]) != 3:
            default_plot_kwargs["title_quantiles"] = None
            default_plot_kwargs["show_titles"] = False

        fig_ = corner.corner(flat_samples, range = range, **default_plot_kwargs)

        # plot fiducial values
        if fid_colour is not None:
            corner.overplot_lines(fig_, self.fiducial_params, color = fid_colour)
            corner.overplot_points(fig_, np.array(self.fiducial_params)[None], marker = "s", color = fid_colour)
        # calculate best fit values of the variables and their associated errors
        # means = []
        # l1_errs = []
        # u1_errs = []
        # for i in range(flat_samples.shape[1]):
        #     mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        #     q = np.diff(mcmc)
        #     means = np.append(means, mcmc[1])
        #     l1_errs = np.append(l1_errs, q[0])
        #     u1_errs = np.append(u1_errs, q[1])
            
        # save true variables
        #self.save(means, l1_errs, u1_errs)
        
        #plt.show()
        # save plot

        if legend:
            # Add a legend in the upper-right corner
            handles = [
                plt.Line2D([], [], color="blue", label="Dataset 1"),
                plt.Line2D([], [], color="red", linestyle="dashed", label="Dataset 2"),
            ]
            fig_.legend(
                handles=handles,
                loc="upper right",
                fontsize=12,
                frameon=True,
                #title="Legend",
            )
            # Adjust spacing to fit the legend
            fig_.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.1, wspace=0.1)

        if save:
            save_path = self.backend_filename.replace(".h5", "_MCMC.jpeg")
            fig_.savefig(save_path, dpi = 600, bbox_inches = "tight")
            galfind_logger.info(f"Saved corner plot to {save_path}")
        return fig_
    
    # def get_residuals(self: Self, params: Dict[str, float]) -> NDArray[float]:
    #     return self.y_data - self.model(self.x_data, params)

    # def sigma_clip(
    #     self: Self,
    #     sigma: float = 3.0
    # ) -> NDArray[bool]:
    #     removed_residuals = sigma_clip(self.get_residuals(self.get_params_med()), sigma = sigma, masked = True)
    #     kept_residuals = ~removed_residuals.mask
    #     return kept_residuals


class MCMC_Fitter(Base_MCMC_Fitter):
    """`Base_MCMC_Fitter` subclass assuming (possibly asymmetric) Gaussian y-errors.

    Implements `log_likelihood` as the sum of the prior log-probability and
    a Gaussian likelihood on the residuals, selecting the appropriate
    upper/lower `y_data_errs` value for each residual via `_get_sigma_sq`.
    Also adds utilities for computing scatter (`get_scatter`,
    `get_empirical_scatter`), summarising the posterior (`get_params_med`,
    `get_params_errs`), sigma-clipping outliers, and saving/loading the
    fitter to/from HDF5. Uses the same constructor as `Base_MCMC_Fitter`.
    Subclasses must still implement `update` and `model`.
    """

    # def _instantiate_walkers(self: Self) -> NoReturn:
    #     # # uniformly distributed starting positions over flat prior
    #     # flat_priors_lower = [self.flat_priors[i][0] for i in range(self.ndim)]
    #     # flat_priors_diff = [self.flat_priors[i][1] - self.flat_priors[i][0] for i in range(self.ndim)]

    #     # pos = [flat_priors_lower + (flat_priors_diff) \
    #     #        * np.random.uniform(0, 1, self.ndim) for i in range(self.nwalkers)]

    #     init_pos = [self.fiducial_params + 1e-4 * np.random.uniform(0, 1, self.ndim) * \
    #             self.fiducial_params for i in range(self.nwalkers)]
    #     # init_pos = [np.array([np.random.uniform(prior.prior_params["lower_lim"], \
    #     #     prior.prior_params["upper_lim"], 1)[0] for prior in self.priors]) \
    #     #     for i in range(self.nwalkers)]
    #     self.init_pos = init_pos

    # def __call__(
    #     self: Self,
    #     n_steps: int,
    #     n_processes: int = 1,
    # ) -> NoReturn:
    #     if hasattr(self, "backend"):
    #         galfind_logger.info("Initial size: {0}".format(self.backend.iteration))
    #         n_steps -= self.backend.iteration
    #     else:
    #         galfind_logger.info("Initial size: 0")
    #     with mp.Pool(processes=n_processes) as pool:
    #         # blobs_dtype = [(key, object) for key in self.blob_keys]
    #         # print("blobs_dtype = ", blobs_dtype)
    #         self.sampler.run_mcmc(self.init_pos, n_steps, progress = True)
    #         if hasattr(self, "backend"):
    #             galfind_logger.info("Final size: {0}".format(self.backend.iteration))
    #     pool.close()

    @abstractmethod
    def update(self):
        """Update any derived/cached fitter state (e.g. after loading a chain).

        Abstract method to be implemented by subclasses.
        """
        pass

    @abstractmethod
    def model(
        self: Self,
        x: Union[float, List[float], NDArray[float]],
        params: Dict[str, float],
        **kwargs: Dict[str, Any],
    ) -> float:
        """Evaluate the model at ``x`` for the given parameters.

        Abstract method to be implemented by subclasses.

        Parameters
        ----------
        x : `float`, `list` of `float`, or `NDArray` of `float`
            Independent variable value(s) at which to evaluate the model.
        params : `dict` of `str`: `float`
            Model parameter values, keyed by parameter name.
        **kwargs : `dict`
            Additional model-specific keyword arguments.

        Returns
        -------
        `float`
            The model value(s) evaluated at ``x``.
        """
        pass

    def _get_sigma_sq(
        self: Self,
        residuals: NDArray[float],
        params: Dict[str, float],
    ) -> NDArray[float]:
        sigma = [y_err_l1 if residual > 0. else y_err_u1 for residual, y_err_l1, y_err_u1 in 
            zip(residuals, self.y_data_errs[0], self.y_data_errs[1])]
        return np.array(sigma) ** 2
    
    def log_likelihood(
        self: Self,
        params: List[float],
    ) -> float:
        """Compute the log-posterior (prior + Gaussian likelihood) for ``params``.

        Combines the summed log-prior probability of `params` with a
        Gaussian log-likelihood of the residuals between `y_data` and
        `model`, using `_get_sigma_sq` for the per-point variance.

        Parameters
        ----------
        params : `list` of `float`
            Free parameter values, in the same order as `priors`.

        Returns
        -------
        `float`
            ``-numpy.inf`` if any parameter falls outside its prior,
            otherwise the log-prior plus Gaussian log-likelihood.
        """
        params_loc = {}
        for i, prior in enumerate(self.priors):
            params_loc[prior.name] = params[i]
        lp = self.priors(params_loc)
        if not np.isfinite(lp):
            return -np.inf # np.full(1 + len(self.blob_keys), -np.inf)
        # update params with fixed values
        params_loc = self._fix_params(params_loc)
        residuals = self.get_residuals(params_loc)
        sigma_sq = self._get_sigma_sq(residuals, params_loc)
        return lp - 0.5 * np.sum(residuals ** 2 / sigma_sq + np.log(sigma_sq))
    
        # if len(self.blob_keys) == 0:
        #     return return_value
        # else:
        #     # append log likelihood to front of blobs array and change to tuple
        #     output_values = tuple(np.insert(self.blob_funcs(params_loc), 0, return_value))
        #     # print(return_value, " ".join(map(str, self.blob_funcs(params_loc))))
        #     # z = (self.blob_funcs(params_loc)[i] for i in range(len(self.blob_keys)))
        #     # x = tuple(self.blob_funcs(params_loc))
        #     # print(x)
        #     # print(type(x))
        #     #print(val + "," for val in self.blob_funcs(params_loc))
        #     return output_values #return_value, " ".join(map(str, self.blob_funcs(params_loc)))

    # def _fix_params(
    #     self: Self,
    #     params: Dict[str, float]
    # ) -> Dict[str, float]:
    #     for key, val in self.fixed_params.items():
    #         params[key] = val
    #     return params
    
    def get_scatter(
        self: Self,
        x_arr: Optional[float] = None,
        **kwargs,
    ) -> float:
        """Compute the intrinsic scatter of the fit, according to `scatter_type`.

        Behaviour depends on the ``self.scatter_type`` attribute (set by
        subclasses, e.g. `Linear_Fitter`): ``"asymmetric"`` computes the
        scatter separately above/below the median-parameter model from the
        residuals; ``"scatter"`` reads the fitted ``"scatter"`` parameter;
        ``"scatter_y"``, ``"scatter_xy"`` and ``"exp_scatter_y"`` compute a
        rolling-average scatter above/below the model as a function of
        `x_arr`.

        Parameters
        ----------
        x_arr : `float`, optional
            x-values at which to evaluate the scatter, required for the
            ``"scatter_y"``, ``"scatter_xy"`` and ``"exp_scatter_y"``
            scatter types. Default is `None`.
        **kwargs : `dict`
            Additional keyword arguments passed on to `get_residuals`.

        Returns
        -------
        `float` or `list`
            The computed scatter. For ``"scatter_y"``, ``"scatter_xy"`` and
            ``"exp_scatter_y"`` this is a ``(y_interp_below, y_interp_above)``
            tuple interpolated onto `x_arr`; otherwise a two-element
            ``[negative, positive]`` scatter list, or `None` if
            `scatter_type` is not recognised.
        """
        if self.scatter_type == "asymmetric":
            residuals = self.get_residuals(self.get_params_med(), **kwargs)
            positive_residuals = residuals[residuals > 0]
            negative_residuals = residuals[residuals < 0]
            positive_scatter = np.sqrt(np.sum(positive_residuals ** 2) / (len(positive_residuals) - 1))
            negative_scatter = np.sqrt(np.sum(negative_residuals ** 2) / (len(negative_residuals) - 1))
            scatter = [positive_residuals, negative_residuals]
            galfind_logger.info(f"Scatter: [+{positive_scatter:.3f}, -{negative_scatter:.3f}] dex")
        elif self.scatter_type == "scatter":
            #residuals = self.get_residuals(self.get_params_med(), **kwargs)
            #scatter = np.sqrt(np.sum(residuals ** 2) / (len(self.y_data) - 1))
            scatter_ = self.get_params_med()["scatter"]
            scatter = [scatter_, scatter_]
            galfind_logger.info(f"Scatter: {scatter_:.3f} dex")
        elif self.scatter_type in ["scatter_y", "scatter_xy", "exp_scatter_y"]:
            med_params = self.get_params_med()
            residuals = self.get_residuals(med_params, **kwargs)
            sorted_indices = np.argsort(self.x_data)
            x_sorted = self.x_data[sorted_indices]
            y_sorted = self.y_data[sorted_indices]
            residuals = residuals[sorted_indices]
            x_sorted_above = x_sorted[residuals > 0]
            x_sorted_below = x_sorted[residuals < 0]
            y_sorted_above = y_sorted[residuals > 0]
            y_sorted_below = y_sorted[residuals < 0]
            if self.scatter_type == "scatter_y":
                y_above_sigma = np.abs(med_params["scatter_y_a"] + y_sorted_above * med_params["scatter_y_b"])
                y_below_sigma = np.abs(med_params["scatter_y_a"] + y_sorted_below * med_params["scatter_y_b"])
            elif self.scatter_type == "scatter_xy":
                y_above_sigma = np.abs(med_params["scatter_c"] + x_sorted_above * med_params["scatter_x"] + \
                    y_sorted_above * med_params["scatter_y"])
                y_below_sigma = np.abs(med_params["scatter_c"] + x_sorted_below * med_params["scatter_x"] + \
                    y_sorted_below * med_params["scatter_y"])
            elif self.scatter_type == "exp_scatter_y":
                y_above_sigma = np.log1p(np.exp(med_params["exp_scatter_y_a"] + ((y_sorted_above - self.y_data.mean()) / self.y_data.std()) * med_params["exp_scatter_y_b"]))
                y_below_sigma = np.log1p(np.exp(med_params["exp_scatter_y_a"] + ((y_sorted_below - self.y_data.mean()) / self.y_data.std()) * med_params["exp_scatter_y_b"]))
            else:
                raise ValueError(f"Unknown scatter type {self.scatter_type=}, cannot plot!")
            window_above = len(y_above_sigma)
            window_below = len(y_below_sigma)
            y_roll_above = funcs.rolling_average(y_above_sigma, window_above)
            y_roll_below = funcs.rolling_average(y_below_sigma, window_below)
            x_roll_above = x_sorted_above[(window_above//2):-(window_above//2)] if window_above % 2 == 1 else x_sorted_above[(window_above//2 - 1):-(window_above//2)]
            x_roll_below = x_sorted_below[(window_below//2):-(window_below//2)] if window_below % 2 == 1 else x_sorted_below[(window_below//2 - 1):-(window_below//2)]
            # interpolate onto same x_arr
            y_interp_below = np.interp(x_arr, x_roll_below, y_roll_below)
            y_interp_above = np.interp(x_arr, x_roll_above, y_roll_above)
            return y_interp_below, y_interp_above
        else:
            scatter = None
            galfind_logger.warning(
                f"{self.scatter_type=} must be in ['asymmetric', 'symmetric', 'scatter_y', 'scatter_xy']. Skipping!"
            )
        self.scatter = scatter
        return self.scatter
    
    def get_empirical_scatter(
        self: Self,
        n_bins: int = 10,
    ) -> None:
        """Compute the empirical scatter of the data about the fitted model.

        Bins `x_data` into `n_bins` bins, and in each bin computes the
        weighted 16th/50th/84th percentiles of `y_data` (weighted by the
        inverse-variance of the appropriate `y_data_errs` side relative to
        the model), taking the median across bins of the resulting
        upper/lower scatter estimates.

        Parameters
        ----------
        n_bins : `int`, optional
            Number of bins to use across the range of `x_data`. Default is
            ``10``.

        Returns
        -------
        `tuple` of `float`
            ``(scat_lo, scat_up)``: the median empirical lower and upper
            scatter estimates across bins.
        """
        from statsmodels.stats.weightstats import DescrStatsW
        
        model_y = self.model(self.x_data, self.params_med, subcls = "sfg")
        bins = np.linspace(self.x_data.min(), self.x_data.max(), n_bins) #int(len(x_arr) / 10_000))
        bin_indices = np.digitize(self.x_data, bins)
        bin_centres = 0.5 * (bins[1:] + bins[:-1])
        plot_bin_centres = []
        model_value_diff = []
        scat_up = []
        scat_lo = []
        Ngals = []
        for i in tqdm(range(1, len(bins)), desc = "Calculating scatter in bins", total=len(bins)-1):
            bin_y = self.y_data[bin_indices == i]
            bin_model_y = model_y[bin_indices == i]
            bin_y_errs_l1 = self.y_data_errs[0][bin_indices == i]
            bin_y_errs_u1 = self.y_data_errs[1][bin_indices == i]
            if len(bin_y) == 0:
                continue
            bin_y_errs = np.array([
                y_l1_ if y_ >= model_y_ else y_u1_ for y_, model_y_, y_l1_, y_u1_
                in zip(bin_y, bin_model_y, bin_y_errs_l1, bin_y_errs_u1)
            ])
            #bin_y_above = bin_y[bin_y >= bin_model_y]
            #bin_y_below = bin_y[bin_y < bin_model_y]
            ds = DescrStatsW(bin_y, weights = 1 / (bin_y_errs ** 2))
            quantiles = ds.quantile([0.16, 0.5, 0.84], return_pandas = False)
            med_model = np.median(bin_model_y)
            #med_values = np.median(bin_y)
            model_value_diff.append(med_model - quantiles[1])
            #print(f"Bin {i}: median model = {med_model}, median values = {med_values}, N = {len(bin_y)}")
            plot_bin_centres.append(bin_centres[i-1])
            sigma_l1_emp = quantiles[1] - quantiles[0] #med_model - np.percentile(bin_y, 16)
            sigma_u1_emp = quantiles[2] - quantiles[1] #np.percentile(bin_y, 84) - med_model
            #print(np.mean(bin_y_errs_u1[bin_y >= bin_model_y]), np.mean(bin_y_errs_l1[bin_y < bin_model_y]))
            scat_up_ = sigma_u1_emp #** 2 - np.mean(bin_y_errs_u1[bin_y >= bin_model_y]) ** 2
            scat_lo_ = sigma_l1_emp #** 2 - np.mean(bin_y_errs_l1[bin_y < bin_model_y]) ** 2
            scat_up.append(scat_up_)
            scat_lo.append(scat_lo_)
            Ngals.append(len(bin_y))
        Ngals = np.array(Ngals)
        ds = DescrStatsW(scat_up) #, weights = 1 / Ngals)
        scat_up_ = ds.quantile([0.5], return_pandas = False)[0]
        ds = DescrStatsW(scat_lo) #, weights = 1 / Ngals)
        scat_lo_ = ds.quantile([0.5], return_pandas = False)[0]
        print(f"Empirical scatter: +{scat_up_}, -{scat_lo_}")
        return scat_lo_, scat_up_
    
    # def get_sample(self: Self) -> NDArray[float]:
    #     autocorr_time = np.max(self.sampler.get_autocorr_time())
    #     discard = int(autocorr_time * 2)
    #     thin = int(autocorr_time / 2)
    #     chain = self.sampler.get_chain(flat = True, discard = discard, thin = thin)
    #     return chain

    def get_params_med(self: Self) -> Dict[str, float]:
        """Return (and cache) the per-parameter posterior median.

        Computed once from `get_sample` and cached as ``self.params_med``
        on subsequent calls.

        Returns
        -------
        `dict` of `str`: `float`
            Median posterior value of each parameter, keyed by prior name.
        """
        if not hasattr(self, "params_med"):
            chain = self.get_sample()
            self.params_med = {prior.name: np.median(chain[:, i]) for i, prior in enumerate(self.priors)}
        galfind_logger.info(f"Median parameters: {self.params_med}")
        return self.params_med

    def get_params_errs(self: Self) -> NDArray[float]:
        """Return (and cache) the per-parameter posterior 1-sigma errors.

        Computed once from `get_sample` (16th/50th/84th percentiles) and
        cached as ``self.params_errs`` on subsequent calls.

        Returns
        -------
        `dict` of `str`: `NDArray` of `float`
            For each parameter, a two-element array
            ``[median - 16th percentile, 84th percentile - median]``.
        """
        if not hasattr(self, "params_errs"):
            chain = self.get_sample()
            params = {prior.name: np.percentile(chain[:, i], [16, 50, 84]) for i, prior in enumerate(self.priors)}
            self.params_errs = {}
            for key, vals in params.items():
                self.params_errs[key] = np.array([vals[1] - vals[0], vals[2] - vals[1]])
        galfind_logger.info(f"{self.params_errs}=")
        return self.params_errs

    def get_residuals(self: Self, params: Dict[str, float]) -> NDArray[float]:
        """Compute the residuals between `y_data` and the model.

        Parameters
        ----------
        params : `dict` of `str`: `float`
            Parameter values passed to `model`.

        Returns
        -------
        `NDArray` of `float`
            ``y_data - model(x_data, params)``.
        """
        return self.y_data - self.model(self.x_data, params)

    def sigma_clip(
        self: Self,
        sigma: float = 3.0
    ) -> NDArray[bool]:
        """Identify data points to keep after sigma-clipping the residuals.

        Residuals are computed at the median posterior parameters
        (`get_params_med`) and sigma-clipped with `astropy.stats.sigma_clip`.

        Parameters
        ----------
        sigma : `float`, optional
            The number of standard deviations to use for the clipping
            limit. Default is ``3.0``.

        Returns
        -------
        `NDArray` of `bool`
            Boolean mask, `True` for data points kept (not clipped).
        """
        removed_residuals = sigma_clip(self.get_residuals(self.get_params_med()), sigma = sigma, masked = True)
        kept_residuals = ~removed_residuals.mask
        return kept_residuals

    @classmethod
    def from_h5(cls: Type[Self], h5_path: str) -> Self:
        """Construct a fitter instance from a saved fitter HDF5 file.

        Reconstructs the `Priors`, data arrays, and sampler settings that
        were previously written by `save_h5`.

        Parameters
        ----------
        h5_path : `str`
            Path to the fitter HDF5 file (as produced by `save_h5`).

        Returns
        -------
        `Self`
            A new fitter instance built from the saved priors and data.
            Note: ``fixed_params`` is not yet restored from file and is
            currently always set to ``{}``.
        """
        galfind_logger.warning(
            "Loading MCMC fitter from HDF5 is not implemented yet, if it works its a hack to get working for thesis!"
        )
        # construct priors from saved .h5 file
        h5_file = h5py.File(h5_path, "r")
        h5_priors = h5_file["priors"]
        prior_types = {prior.__name__: prior for prior in Prior.__subclasses__()}
        priors = Priors(
            [
                prior_types[str(h5_priors[prior_name]["prior_type"][()].decode("utf-8"))](
                    prior_name, 
                    [
                        float(h5_priors[prior_name]["lower_lim"][()]),
                        float(h5_priors[prior_name]["upper_lim"][()]),
                    ],
                    float(h5_priors[prior_name]["fiducial"][()])
                )
                for prior_name in [name.decode("utf-8") for name in h5_file["priors_names"][()]]
            ]
        )
        # ensure these priors are the correct way round
        x_data = h5_file["x_data"][()]
        y_data = h5_file["y_data"][()]
        y_data_errs = h5_file["y_data_errs"][()]
        n_walkers = h5_file["nwalkers"][()]
        backend_filename = h5_file["backend_filename"][()].decode("utf-8")
        #init_pos = h5_file["init_pos"][()]
        # TODO: load these in
        #incl_scatter = h5_file["incl_scatter"][()]
        #fixed_params = json.loads(h5_file["fixed_params"][()])#.decode("utf-8"))
        h5_file.close()
        return cls(priors, x_data, y_data, y_data_errs, n_walkers, backend_filename = backend_filename, fixed_params = {}) #, incl_scatter = incl_scatter)#, init_pos = init_pos)
    
    def save_h5(self: Self) -> str:
        """Save the priors, data, and sampler settings of this fitter to HDF5.

        Writes ``backend_filename`` with ``.h5`` replaced by ``_fitter.h5``,
        storing the priors (name, limits, fiducial, type), `x_data`,
        `y_data`, `y_data_errs`, `nwalkers`, `fixed_params` and
        `backend_filename`, such that the fitter can later be reconstructed
        with `from_h5`.

        Returns
        -------
        `str`
            Path to the saved HDF5 file.
        """
        # open h5 file
        h5_out_name = self.backend_filename.replace(".h5", "_fitter.h5")
        out_file = h5py.File(h5_out_name, "w")
        priors = out_file.create_group("priors")
        for prior in self.priors:
            prior_ = priors.create_group(prior.name)
            prior_["lower_lim"] = prior.prior_params["lower_lim"]
            prior_["upper_lim"] = prior.prior_params["upper_lim"]
            prior_["fiducial"] = prior.fiducial
            prior_["prior_type"] = prior.__class__.__name__
        priors_names = [prior.name for prior in self.priors]
        out_file.create_dataset("priors_names", data = np.array(priors_names, dtype = "S"))
        out_file.create_dataset("x_data", data = self.x_data)
        out_file.create_dataset("y_data", data = self.y_data)
        out_file.create_dataset("y_data_errs", data = self.y_data_errs)
        out_file.create_dataset("nwalkers", data = self.nwalkers)
        out_file.create_dataset("fixed_params", data = json.dumps(self.fixed_params))
        #out_file.create_dataset("incl_scatter", data = self.incl_scatter)
        out_file.create_dataset("backend_filename", data = self.backend_filename)
        out_file.close()
        galfind_logger.info(f"Saved MCMC fitter to {h5_out_name}")
        return h5_out_name

# class Scattered_MCMC_Fitter(Base_MCMC_Fitter):

#     def __init__(
#         self: Self,
#         priors: Priors,
#         x_data_chains: NDArray[NDArray[float]],
#         y_data_chains: NDArray[NDArray[float]],
#         nwalkers: int,
#         backend_filename: Optional[str],
#         fixed_params: Dict[str, float],
#     ):
#         assert "scatter" in priors.names + list(fixed_params.keys())
#         y_data_errs = None
#         super().__init__(priors, x_data_chains, y_data_chains, y_data_errs, nwalkers, backend_filename, fixed_params)
#         self.mask = self.x_data.mask | self.y_data.mask

#     @abstractmethod
#     def update(self):
#         pass
    
#     @abstractmethod
#     def model(
#         self: Self,
#         x: Union[float, List[float], NDArray[float]],
#         params: Dict[str, float]
#     ) -> float:
#         pass

#     def _get_plot_chains(
#         self: Self,
#         x_arr: NDArray[float],
#         log_data: bool = False,
#     ) -> Tuple[NDArray[float], NDArray[float], NDArray[float]]:
#         autocorr_time = np.max(self.sampler.get_autocorr_time())
#         discard = int(autocorr_time * 2)
#         thin = 1 # int(autocorr_time / 2)
#         if log_data:
#             model = lambda x, params: np.log10(self.model(x, params))
#         else:
#             model = self.model
#         params_arr = [{prior.name: param for prior, param in zip(self.priors, params)} \
#             for params in self.sampler.get_chain(flat = True, discard = discard, thin = thin)]
#         y_fit = np.array([model(x_arr, params) for params in params_arr]).T
#         l1_chains = np.array([np.percentile(y, 16) for y in y_fit])
#         med_chains = np.array([np.percentile(y, 50) for y in y_fit])
#         u1_chains = np.array([np.percentile(y, 84) for y in y_fit])
#         return l1_chains, med_chains, u1_chains

#     def log_likelihood(
#         self: Self,
#         params: List[float],
#     ) -> float:
#         params_loc = {}
#         for i, prior in enumerate(self.priors):
#             params_loc[prior.name] = params[i]
#         lp = self.priors(params_loc)
#         if not np.isfinite(lp):
#             return -np.inf # np.full(1 + len(self.blob_keys), -np.inf)
#         else:
#             model = self.model(self.x_data, params_loc)
#             model.mask = self.mask
#             probs_samples = stats.norm(model, params_loc["scatter"]).pdf(self.y_data)
#             probs_objects = np.nanmean(probs_samples, axis = 1)
#             assert len(probs_objects) == len(self.x_data)
#             return np.log(probs_objects + 1e-100).sum()


class Schechter_Lum_Fitter(MCMC_Fitter):
    """`MCMC_Fitter` implementing a Schechter luminosity function.

    Fits ``L_star``, ``alpha`` and ``log10_phi_star``, any of which (except
    ``log10_phi_star``) may instead be held fixed via `fixed_params`.

    Parameters
    ----------
    priors : `Priors`
        Priors for the free parameters among ``L_star``, ``alpha`` and
        ``log10_phi_star`` (whichever are not in `fixed_params`).
    x_data : `NDArray` of `float`
        Luminosity values.
    y_data : `NDArray` of `float`
        Number density values.
    y_data_errs : `NDArray` of `NDArray` of `float`
        Lower/upper errors on `y_data`.
    nwalkers : `int`
        Number of MCMC walkers.
    backend_filename : `str` or `None`
        Path to an HDF5 backend file for the sampler, or `None`.
    fixed_params : `dict` of `str`: `float`
        Fixed values for ``L_star`` and/or ``alpha``. Must, together with
        `priors`, contain exactly ``L_star``, ``alpha`` and
        ``log10_phi_star``.
    """

    def __init__(
        self: Self,
        priors: Priors,
        x_data: NDArray[float],
        y_data: NDArray[float],
        y_data_errs: NDArray[NDArray[float, float]],
        nwalkers: int,
        backend_filename: Optional[str],
        fixed_params: Dict[str, float]
    ):
        assert all([key in ["L_star", "alpha"] for key in fixed_params.keys()]), \
            galfind_logger.critical(
                f"{fixed_params=} must be L_star and/or alpha or empty"
            )
        assert len(fixed_params) + len(priors) == 3, \
            galfind_logger.critical(
                "Must have exactly 3 parameters"
            )
        assert all([key in fixed_params.keys() or key in priors.names for key in ["L_star", "alpha", "log10_phi_star"]]), \
            galfind_logger.critical(
                f"{repr(priors)=} or {fixed_params=} must contain L_star, alpha, and log10_phi_star"
            )
        super().__init__(
            priors, 
            x_data, 
            y_data, 
            y_data_errs, 
            nwalkers, 
            backend_filename,
            fixed_params
        )
    
    def model(
        self: Self,
        x: Union[float, List[float], NDArray[float]],
        params: Dict[str, float],
        **kwargs: Dict[str, Any],
    ) -> float:
        """Evaluate the Schechter luminosity function.

        Parameters
        ----------
        x : `float`, `list` of `float`, or `NDArray` of `float`
            Luminosity value(s) at which to evaluate the function.
        params : `dict` of `str`: `float`
            Must contain ``"log10_phi_star"``, ``"L_star"`` and ``"alpha"``.
        **kwargs : `dict`
            Unused, present for interface compatibility.

        Returns
        -------
        `float`
            The Schechter function number density evaluated at ``x``.
        """
        return 10 ** params["log10_phi_star"] * ((x / params["L_star"]) ** params["alpha"]) * \
            np.exp(-x / params["L_star"]) / params["L_star"]

    def update(
        self: Self
    ) -> NoReturn:
        """No-op update; this fitter has no derived state to refresh."""
        pass


class Schechter_Mag_Fitter(MCMC_Fitter):
    """`MCMC_Fitter` implementing a Schechter function in magnitude space.

    Fits ``M_star``, ``alpha`` and ``log10_phi_star``, any of which (except
    ``log10_phi_star``) may instead be held fixed via `fixed_params`.

    Parameters
    ----------
    priors : `Priors`
        Priors for the free parameters among ``M_star``, ``alpha`` and
        ``log10_phi_star`` (whichever are not in `fixed_params`).
    x_data : `NDArray` of `float`
        Absolute magnitude values.
    y_data : `NDArray` of `float`
        Number density values.
    y_data_errs : `NDArray` of `NDArray` of `float`
        Lower/upper errors on `y_data`.
    nwalkers : `int`
        Number of MCMC walkers.
    backend_filename : `str` or `None`
        Path to an HDF5 backend file for the sampler, or `None`.
    fixed_params : `dict` of `str`: `float`
        Fixed values for ``M_star`` and/or ``alpha``. Must, together with
        `priors`, contain exactly ``M_star``, ``alpha`` and
        ``log10_phi_star``.
    """

    def __init__(
        self: Self,
        priors: Priors,
        x_data: NDArray[float],
        y_data: NDArray[float],
        y_data_errs: NDArray[NDArray[float, float]],
        nwalkers: int,
        backend_filename: Optional[str],
        fixed_params: Dict[str, float]
    ):
        assert all([key in ["M_star", "alpha"] for key in fixed_params.keys()]), \
            galfind_logger.critical(
                f"{fixed_params=} must be M_star and/or alpha or empty"
            )
        assert len(fixed_params) + len(priors) == 3, \
            galfind_logger.critical(
                "Must have exactly 3 parameters"
            )
        assert all([key in fixed_params.keys() or key in priors.names for key in ["M_star", "alpha", "log10_phi_star"]]), \
            galfind_logger.critical(
                f"{repr(priors)=} or {fixed_params=} must contain M_star, alpha, and log10_phi_star"
            )
        super().__init__(
            priors, 
            x_data, 
            y_data, 
            y_data_errs, 
            nwalkers, 
            backend_filename,
            fixed_params
        )
    
    def model(
        self: Self,
        x: Union[float, List[float], NDArray[float]],
        params: Dict[str, float],
        **kwargs: Dict[str, Any],
    ) -> float:
        """Evaluate the Schechter function in magnitude space.

        Parameters
        ----------
        x : `float`, `list` of `float`, or `NDArray` of `float`
            Absolute magnitude value(s) at which to evaluate the function.
        params : `dict` of `str`: `float`
            Must contain ``"log10_phi_star"``, ``"M_star"`` and ``"alpha"``.
        **kwargs : `dict`
            Unused, present for interface compatibility.

        Returns
        -------
        `float`
            The Schechter function number density evaluated at ``x``.
        """
        return (10 ** params["log10_phi_star"]) * 0.4 * np.log(10) * \
            10 ** (0.4 * (params["alpha"] + 1) * (params["M_star"] - x)) * \
            np.exp(-10 ** (0.4 * (params["M_star"] - x)))

    def update(
        self: Self
    ) -> NoReturn:
        """No-op update; this fitter has no derived state to refresh."""
        pass

class DPL_Lum_Fitter(MCMC_Fitter):
    """`MCMC_Fitter` implementing a double power-law luminosity function.

    Fits ``L_star``, ``alpha``, ``beta`` and ``log10_phi_star``, any of
    which (except ``log10_phi_star``) may instead be held fixed via
    `fixed_params`.

    Parameters
    ----------
    priors : `Priors`
        Priors for the free parameters among ``L_star``, ``alpha``,
        ``beta`` and ``log10_phi_star`` (whichever are not in
        `fixed_params`).
    x_data : `NDArray` of `float`
        Luminosity values.
    y_data : `NDArray` of `float`
        Number density values.
    y_data_errs : `NDArray` of `NDArray` of `float`
        Lower/upper errors on `y_data`.
    nwalkers : `int`
        Number of MCMC walkers.
    backend_filename : `str` or `None`
        Path to an HDF5 backend file for the sampler, or `None`.
    fixed_params : `dict` of `str`: `float`
        Fixed values for ``L_star``, ``alpha`` and/or ``beta``. Must,
        together with `priors`, contain exactly ``L_star``, ``alpha``,
        ``beta`` and ``log10_phi_star``.
    """

    def __init__(
        self: Self,
        priors: Priors,
        x_data: NDArray[float],
        y_data: NDArray[float],
        y_data_errs: NDArray[NDArray[float, float]],
        nwalkers: int,
        backend_filename: Optional[str],
        fixed_params: Dict[str, float]
    ):
        assert all([key in ["L_star", "alpha", "beta"] for key in fixed_params.keys()]), \
            galfind_logger.critical(
                f"{fixed_params=} must be L_star, alpha and/or beta, or empty"
            )
        assert len(fixed_params) + len(priors) == 4, \
            galfind_logger.critical(
                "Must have exactly 4 parameters"
            )
        assert all([key in fixed_params.keys() or key in priors.names \
                for key in ["L_star", "alpha", "beta", "log10_phi_star"]]), \
            galfind_logger.critical(
                f"{repr(priors)=} or {fixed_params=} must contain L_star, alpha, beta, and log10_phi_star"
            )
        super().__init__(priors, x_data, y_data, y_data_errs, nwalkers, backend_filename, fixed_params)
    
    def model(
        self: Self,
        x: Union[float, List[float], NDArray[float]],
        params: Dict[str, float],
        **kwargs: Dict[str, Any],
    ) -> float:
        """Evaluate the double power-law luminosity function.

        Parameters
        ----------
        x : `float`, `list` of `float`, or `NDArray` of `float`
            Luminosity value(s) at which to evaluate the function.
        params : `dict` of `str`: `float`
            Must contain ``"log10_phi_star"``, ``"L_star"``, ``"alpha"``
            and ``"beta"``.
        **kwargs : `dict`
            Unused, present for interface compatibility.

        Returns
        -------
        `float`
            The double power-law number density evaluated at ``x``.
        """
        return ((10 ** params["log10_phi_star"]) / params["L_star"]) * \
            (
                ((x / params["L_star"]) ** params["alpha"]) + \
                ((x / params["L_star"]) ** params["beta"])
            ) ** -1.0

    def update(
        self: Self
    ) -> NoReturn:
        """No-op update; this fitter has no derived state to refresh."""
        pass


class DPL_Mag_Fitter(MCMC_Fitter):
    """`MCMC_Fitter` implementing a double power-law function in magnitude space.

    Fits ``M_star``, ``alpha``, ``beta`` and ``log10_phi_star``, any of
    which (except ``log10_phi_star``) may instead be held fixed via
    `fixed_params`.

    Parameters
    ----------
    priors : `Priors`
        Priors for the free parameters among ``M_star``, ``alpha``,
        ``beta`` and ``log10_phi_star`` (whichever are not in
        `fixed_params`).
    x_data : `NDArray` of `float`
        Absolute magnitude values.
    y_data : `NDArray` of `float`
        Number density values.
    y_data_errs : `NDArray` of `NDArray` of `float`
        Lower/upper errors on `y_data`.
    nwalkers : `int`
        Number of MCMC walkers.
    backend_filename : `str` or `None`
        Path to an HDF5 backend file for the sampler, or `None`.
    fixed_params : `dict` of `str`: `float`
        Fixed values for ``M_star``, ``alpha`` and/or ``beta``. Must,
        together with `priors`, contain exactly ``M_star``, ``alpha``,
        ``beta`` and ``log10_phi_star``.
    """

    def __init__(
        self: Self,
        priors: Priors,
        x_data: NDArray[float],
        y_data: NDArray[float],
        y_data_errs: NDArray[NDArray[float, float]],
        nwalkers: int,
        backend_filename: Optional[str],
        fixed_params: Dict[str, float]
    ):
        assert all([key in ["M_star", "alpha", "beta"] for key in fixed_params.keys()]), \
            galfind_logger.critical(
                f"{fixed_params=} must be M_star, alpha and/or beta, or empty"
            )
        assert len(fixed_params) + len(priors) == 4, \
            galfind_logger.critical(
                "Must have exactly 4 parameters"
            )
        assert all([key in fixed_params.keys() or key in priors.names \
                for key in ["M_star", "alpha", "beta", "log10_phi_star"]]), \
            galfind_logger.critical(
                f"{repr(priors)=} or {fixed_params=} must contain M_star, alpha, beta, and log10_phi_star"
            )
        super().__init__(priors, x_data, y_data, y_data_errs, nwalkers, backend_filename, fixed_params)
    
    def model(
        self: Self,
        x: Union[float, List[float], NDArray[float]],
        params: Dict[str, float],
        **kwargs: Dict[str, Any],
    ) -> float:
        """Evaluate the double power-law function in magnitude space.

        Parameters
        ----------
        x : `float`, `list` of `float`, or `NDArray` of `float`
            Absolute magnitude value(s) at which to evaluate the function.
        params : `dict` of `str`: `float`
            Must contain ``"log10_phi_star"``, ``"M_star"``, ``"alpha"``
            and ``"beta"``.
        **kwargs : `dict`
            Unused, present for interface compatibility.

        Returns
        -------
        `float`
            The double power-law number density evaluated at ``x``.
        """
        numerator = 0.4 * np.log(10) * 10 ** (params["log10_phi_star"]) * \
            10 ** (0.4 * (params["M_star"] - x))
        denominator = 10 ** (-0.4 * params["alpha"] * (params["M_star"] - x)) + \
            10 ** (-0.4 * params["beta"] * (params["M_star"] - x))
        return numerator / denominator

    def update(
        self: Self
    ) -> NoReturn:
        """No-op update; this fitter has no derived state to refresh."""
        pass


class Linear_Fitter(MCMC_Fitter):
    """`MCMC_Fitter` implementing a linear model ``y = m * x + c``.

    Fits ``m`` and ``c`` (unless held fixed via `fixed_params`), and
    optionally an additional intrinsic scatter model, whose type is
    inferred automatically from which scatter-related parameter names are
    present in `priors`/`fixed_params`: none (no scatter), ``"scatter"``
    (single symmetric scatter), ``"scatter_up"``/``"scatter_lo"``
    (asymmetric scatter), ``"scatter_y_a"``/``"scatter_y_b"``
    (y-dependent scatter), ``"exp_scatter_y_a"``/``"exp_scatter_y_b"``
    (softplus y-dependent scatter), or ``"scatter_c"``/``"scatter_x"``/
    ``"scatter_y"`` (scatter linear in x and y).

    Parameters
    ----------
    priors : `Priors`
        Priors for the free parameters among ``m``, ``c`` and any scatter
        parameters (whichever are not in `fixed_params`).
    x_data : `NDArray` of `float`
        Independent variable data values.
    y_data : `NDArray` of `float`
        Dependent variable data values.
    y_data_errs : `NDArray` of `NDArray` of `float`
        Lower/upper errors on `y_data`.
    nwalkers : `int`
        Number of MCMC walkers.
    backend_filename : `str`, optional
        Path to an HDF5 backend file for the sampler. Default is `None`.
    fixed_params : `dict` of `str`: `float`, optional
        Fixed values for ``m``, ``c`` and/or any scatter parameters.
        Default is ``{}``.

    Attributes
    ----------
    scatter_type : `str` or `None`
        The inferred type of intrinsic scatter model being fit, or `None`
        if no scatter parameters were provided.
    """
    # Power Law in linear space
    def __init__(
        self: Self,
        priors: Priors,
        x_data: NDArray[float],
        y_data: NDArray[float],
        y_data_errs: NDArray[NDArray[float, float]],
        nwalkers: int,
        backend_filename: Optional[str] = None,
        fixed_params: Dict[str, float] = {},
    ):
        params = ["m", "c"]
        scatter_params = ["scatter", "scatter_up", "scatter_lo", "scatter_f", "scatter_logk", "scatter_y_a", "scatter_y_b", "exp_scatter_y_a", "exp_scatter_y_b", "scatter_c", "scatter_x", "scatter_y"]

        assert all([key in params + scatter_params for key in fixed_params.keys()]), \
            galfind_logger.critical(
                f"{fixed_params=} must be in {','.join(params + scatter_params)}"
            )
        assert all([key in fixed_params.keys() or key in priors.names for key in params]), \
            galfind_logger.critical(
                f"{repr(priors)=} or {fixed_params=} must contain {', '.join(params)}"
            )
        input_scatter_names = [name for name in list(priors.names) + list(fixed_params.keys()) if name in scatter_params]
        if len(input_scatter_names) == 0:
            scatter_type = None
        elif len(input_scatter_names) == 1:
            scatter_type = input_scatter_names[0]
            assert scatter_type in ["scatter"], \
                galfind_logger.critical(
                    f"Only 'scatter' are supported as scatter parameters, got {scatter_type}"
                )
        elif len(input_scatter_names) == 2:
            if all([name in input_scatter_names for name in ["scatter_up", "scatter_lo"]]):
                scatter_type = "scatter_asymmetric"
            elif all([name in input_scatter_names for name in ["scatter_y_a", "scatter_y_b"]]):
                scatter_type = "scatter_y" # NOT GENERAL!
            elif all([name in input_scatter_names for name in ["exp_scatter_y_a", "exp_scatter_y_b"]]):
                scatter_type = "exp_scatter_y"
            else:
                galfind_logger.critical(
                    f"Must have both 'scatter_up' and 'scatter_lo' if using asymmetric scatter, " + \
                    f" both 'scatter_y_a' and 'scatter_y_b', or both 'exp_scatter_y_a' and 'exp_scatter_y_b'" + \
                    f"if using asymmetric scatter, got {input_scatter_names}"
                )
        elif len(input_scatter_names) == 3:
            assert all([name in input_scatter_names for name in ["scatter_c", "scatter_x", "scatter_y"]]), \
                galfind_logger.critical(
                    f"Must have 'scatter_c', 'scatter_x', and 'scatter_y' if using scatter in linear space, " + \
                    f"got {input_scatter_names}"
                )
            scatter_type = "scatter_xy"
        else:
            galfind_logger.critical(
                f"Too many scatter parameters provided: {input_scatter_names}, " + \
                f"{len(input_scatter_names)} > 2, only 'scatter' or 'scatter_up' and 'scatter_lo' are supported!"
            )
        assert len(fixed_params) + len(priors) == len(params) + len(input_scatter_names), \
            galfind_logger.critical(
                f"{len(fixed_params) + len(priors)}!={len(params) + len(input_scatter_names)} required for {scatter_type=}!"
            )
        self.scatter_type = scatter_type
        super().__init__(priors, x_data, y_data, y_data_errs, nwalkers, backend_filename, fixed_params)

    def _get_sigma_sq(
        self: Self,
        residuals: NDArray[float],
        params: Dict[str, float],
    ) -> NDArray[float]:
        if self.scatter_type is None:
            return super()._get_sigma_sq(residuals, params)
        elif self.scatter_type == "scatter":
            return super()._get_sigma_sq(residuals, params) + params["scatter"] ** 2
        elif self.scatter_type == "scatter_asymmetric":
            scatter_arr = np.array([params["scatter_lo"] if residual < 0.0 else params["scatter_up"] for residual in residuals])
            return super()._get_sigma_sq(residuals, params) + scatter_arr ** 2
        elif self.scatter_type == "scatter_-yi":
            return super()._get_sigma_sq(residuals, params) + \
                (params["scatter_f"] * (10 ** - (self.y_data * (10 ** params["scatter_logk"])))) ** 2
        elif self.scatter_type == "scatter_y":
            return super()._get_sigma_sq(residuals, params) + (params["scatter_y_a"] + self.y_data * params["scatter_y_b"]) ** 2
        elif self.scatter_type == "exp_scatter_y":
            # softplus
            return super()._get_sigma_sq(residuals, params) + \
                (np.log1p(np.exp(params["exp_scatter_y_a"] + \
                ((self.y_data - self.y_data.mean()) / self.y_data.std()) * \
                params["exp_scatter_y_b"]))) ** 2
        elif self.scatter_type == "scatter_xy":
            return super()._get_sigma_sq(residuals, params) + \
                (
                    params["scatter_c"] + \
                    self.x_data * params["scatter_x"] + \
                    self.y_data * params["scatter_y"]
                ) ** 2
            

    def model(
        self: Self,
        x: Union[float, List[float], NDArray[float]],
        params: Dict[str, float],
        **kwargs: Dict[str, Any],
    ) -> float:
        """Evaluate the linear model ``y = m * x + c``.

        Parameters
        ----------
        x : `float`, `list` of `float`, or `NDArray` of `float`
            Independent variable value(s) at which to evaluate the model.
        params : `dict` of `str`: `float`
            Must contain ``"m"`` and ``"c"``.
        **kwargs : `dict`
            Unused, present for interface compatibility.

        Returns
        -------
        `float`
            ``params["m"] * x + params["c"]``.
        """
        return params["m"] * x + params["c"]

    def update(
        self: Self
    ) -> NoReturn:
        """No-op update; this fitter has no derived state to refresh."""
        pass

    def plot(
        self: Self,
        ax: plt.Axes,
        log_data: bool = False,
        x_arr: Optional[NDArray[float]] = None,
        plot_scatter: bool = True,
        plot_kwargs: Dict[str, Any] = {},
        fill_between_kwargs: Dict[str, Any] = {},
        scatter_kwargs: Dict[str, Any] = {},
        xoff: float = 0.0,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Plot the median linear fit, its 1-sigma band, and optionally scatter.

        Extends `Base_MCMC_Fitter.plot` by additionally shading a band
        showing the (empirical or fitted, according to `scatter_type`)
        intrinsic scatter about the fit.

        Parameters
        ----------
        ax : `plt.Axes`
            Matplotlib axis to plot the fit on.
        log_data : `bool`, optional
            Whether to plot `model` evaluated in log10 space. Default is
            `False`.
        x_arr : `NDArray` of `float`, optional
            x-values at which to evaluate the model. If `None`, 100
            linearly spaced values spanning `x_data` are used. Default is
            `None`.
        plot_scatter : `bool`, optional
            Whether to additionally plot the intrinsic scatter band.
            Default is `True`.
        plot_kwargs : `dict`, optional
            Keyword arguments passed to `ax.plot` for the median line.
            Default is `{}`.
        fill_between_kwargs : `dict`, optional
            Keyword arguments passed to `ax.fill_between` for the 1-sigma
            band. Default is `{}`.
        scatter_kwargs : `dict`, optional
            Keyword arguments passed to `ax.fill_between` for the scatter
            band, overriding the defaults. Default is `{}`.
        xoff : `float`, optional
            Offset subtracted from `x_arr` before plotting. Default is
            ``0.0``.
        **kwargs : `dict`
            Additional keyword arguments (currently unused here, but
            accepted for interface compatibility).

        Returns
        -------
        `tuple`
            ``(x_arr, med_chains, l1_chains, u1_chains)``, or, if
            `plot_scatter` is `True`, additionally the lower/upper scatter
            band values ``(..., l1_chains_scat, u1_chains_scat)``.
        """
        if plot_scatter:
            def_scatter_kwargs = {
                "color": "grey",
                "alpha": 0.5,
                "zorder": 100,
                "path_effects": [pe.withStroke(linewidth = 2., foreground = "white")],
            }
            for key, val in scatter_kwargs.items():
                def_scatter_kwargs[key] = val
            
            if x_arr is None:
                x_arr = np.linspace(np.min(self.x_data), np.max(self.x_data), 100)
                l1_chains_scat, med_chains, u1_chains_scat = self._get_plot_chains(x_arr = x_arr, log_data = log_data)
            if self.scatter_type is None:
                galfind_logger.info("Plotting empirical scatter not included in MCMC fit")
                scatter_l1, scatter_u1 = self.get_empirical_scatter()
                l1_chains_scat -= scatter_l1
                u1_chains_scat += scatter_u1
            else:
                galfind_logger.info("Plotting fitted scatter of MCMC fit")
                med_params = self.get_params_med()
                if self.scatter_type == "scatter":
                    l1_chains_scat -= med_params["scatter"]
                    u1_chains_scat += med_params["scatter"]
                elif self.scatter_type == "scatter_asymmetric":
                    # deliberately wrong way round, as sign convention in log likelihood also swapped
                    l1_chains_scat -= med_params["scatter_up"]
                    u1_chains_scat += med_params["scatter_lo"]
                elif self.scatter_type in ["scatter_y", "scatter_xy", "exp_scatter_y"]:
                    y_interp_below, y_interp_above = self.get_scatter(x_arr = x_arr)
                    l1_chains_scat -= y_interp_below
                    u1_chains_scat += y_interp_above
                else:
                    err_message = f"Unknown {self.scatter_type=}, cannot plot!"
                    galfind_logger.warning(err_message)
                #ax.plot(x_arr, med_chains, color = colour, zorder = 200, label = label)
                ax.fill_between(x_arr - xoff, l1_chains_scat, u1_chains_scat, **def_scatter_kwargs)
        else:
            galfind_logger.info("Not plotting scatter of MCMC fit")
        
        x_arr, med_chains, l1_chains, u1_chains = super().plot(
            ax,
            log_data = log_data,
            x_arr = x_arr,
            #label = label,
            #colour = colour,
            plot_kwargs = plot_kwargs,
            fill_between_kwargs = fill_between_kwargs,
            xoff = xoff,
        )
        if plot_scatter:
            return x_arr, med_chains, l1_chains, u1_chains, l1_chains_scat, u1_chains_scat
        else:
            return x_arr, med_chains, l1_chains, u1_chains


class Power_Law_Fitter(MCMC_Fitter):
    """`MCMC_Fitter` implementing a power-law model ``y = A * x ** slope``.

    Fits ``A`` and ``slope`` (unless held fixed via `fixed_params`), with
    an optional additional ``logf`` parameter used to inflate the modelled
    variance by a fractional amount of the model value.

    Parameters
    ----------
    priors : `Priors`
        Priors for the free parameters among ``A``, ``slope`` and (if
        `incl_logf` is `True`) ``logf`` (whichever are not in
        `fixed_params`).
    x_data : `NDArray` of `float`
        Independent variable data values.
    y_data : `NDArray` of `float`
        Dependent variable data values.
    y_data_errs : `NDArray` of `NDArray` of `float`
        Lower/upper errors on `y_data`.
    nwalkers : `int`
        Number of MCMC walkers.
    backend_filename : `str` or `None`
        Path to an HDF5 backend file for the sampler, or `None`.
    fixed_params : `dict` of `str`: `float`
        Fixed values for ``A``, ``slope`` and/or ``logf``.
    incl_logf : `bool`, optional
        Whether to include the additional ``logf`` variance-inflation
        parameter. Default is `False`.

    Attributes
    ----------
    incl_logf : `bool`
        Whether the ``logf`` variance-inflation parameter is included.
    """
    # Linear in log-log space
    def __init__(
        self: Self,
        priors: Priors,
        x_data: NDArray[float],
        y_data: NDArray[float],
        y_data_errs: NDArray[NDArray[float, float]],
        nwalkers: int,
        backend_filename: Optional[str],
        fixed_params: Dict[str, float],
        incl_logf: bool = False,
    ):
        self.incl_logf = incl_logf
        params = ["A", "slope"]
        if incl_logf:
            params.append("logf")
        # TODO: Whack these assertions in parent class
        assert all([key in params for key in fixed_params.keys()]), \
            galfind_logger.critical(
                f"{fixed_params=} must be A and/or slope or empty"
            )
        assert len(fixed_params) + len(priors) == len(params), \
            galfind_logger.critical(
                f"Must have exactly {len(params)} parameters if " + \
                f"{'not' if not incl_logf else ''} including logf"
            )
        assert all([key in fixed_params.keys() or key in priors.names for key in params]), \
            galfind_logger.critical(
                f"{repr(priors)=} or {fixed_params=} must contain {', '.join(params)}"
            )
        super().__init__(priors, x_data, y_data, y_data_errs, nwalkers, backend_filename, fixed_params)
    
    def _get_sigma_sq(
        self: Self,
        residuals: NDArray[float],
        params: Dict[str, float]
    ) -> NDArray[float]:
        if self.incl_logf:
            return super()._get_sigma_sq(residuals, params) + \
                np.exp(2 * params["logf"]) * self.model(self.x_data, params) ** 2
        else:
            return super()._get_sigma_sq(residuals, params)

    def model(
        self: Self,
        x: Union[float, List[float], NDArray[float]],
        params: Dict[str, float],
        **kwargs: Dict[str, Any],
    ) -> float:
        """Evaluate the power-law model ``y = A * x ** slope``.

        Parameters
        ----------
        x : `float`, `list` of `float`, or `NDArray` of `float`
            Independent variable value(s) at which to evaluate the model.
        params : `dict` of `str`: `float`
            Must contain ``"A"`` and ``"slope"``.
        **kwargs : `dict`
            Unused, present for interface compatibility.

        Returns
        -------
        `float`
            ``params["A"] * x ** params["slope"]``.
        """
        return params["A"] * x ** params["slope"]

    def update(
        self: Self
    ) -> NoReturn:
        """No-op update; this fitter has no derived state to refresh."""
        pass

    # def plot(
    #     self: Self,
    #     ax: plt.Axes,
    #     log_data: bool = True,
    #     x_arr: Optional[NDArray[float]] = None,
    #     **kwargs: Dict[str, Any]
    # ) -> None:
    #     super().plot(ax, log_data, x_arr, **kwargs)


def get_gal_bias_fitter(
    fitter: Tuple[
        Schechter_Lum_Fitter,
        Schechter_Mag_Fitter,
        DPL_Lum_Fitter,
        DPL_Mag_Fitter
    ],
    sigma_dm_dict: Dict[str, float],
) -> Type[MCMC_Fitter]:

    """
    A factory function to create a Galaxy Bias Fitter class based on the provided fitter class.
    """

    # ensure fitter class is of required type
    fitter_classes = [
        Schechter_Lum_Fitter, Schechter_Mag_Fitter,
        DPL_Lum_Fitter, DPL_Mag_Fitter,
    ]
    assert fitter.__name__ in [subcls.__name__ for subcls in fitter_classes], \
        galfind_logger.critical(
            f"{repr(fitter)=} not in {fitter_classes=}"
        )

    class Galaxy_Bias_Fitter(fitter):
        """MCMC fitter for galaxy bias across multiple surveys.

        Extends the base MCMC fitter to include a galaxy bias parameter that
        is fit independently for each survey, accounting for survey-dependent
        systematic uncertainties.

        Parameters
        ----------
        surveys_arr : `list` of `str`
            List of survey names corresponding to data.
        **kwargs
            Additional keyword arguments passed to base MCMC fitter,
            including 'priors' which must include a 'b_gal' prior.
        """

        def __init__(
            self: Self,
            surveys_arr: List[str],
            **kwargs,
        ):
            assert "priors" in kwargs.keys(), \
                galfind_logger.critical("Must provide priors!")
            assert "b_gal" in kwargs["priors"].names, \
                galfind_logger.critical(
                    f"{kwargs['priors'].names} must include b_gal!"
                )
            #gal_bias_prior = kwargs["priors"]["b_gal"]
            # remove b_gal from priors
            #kwargs["priors"] = Priors([prior for prior in kwargs["priors"] if prior.name != "b_gal"])
            
            Base_MCMC_Fitter.__init__(self, **kwargs)

            #self.priors += gal_bias_prior

            self.surveys_arr = surveys_arr
            assert len(self.surveys_arr) == len(self.x_data) == len(self.y_data) == self.y_data_errs.shape[1], \
                galfind_logger.critical(
                    f"{len(self.surveys_arr)=}, {len(self.x_data)=}, {len(self.y_data)=}, {self.y_data_errs.shape[1]=}, must all be equal!"
                )
            self.sigma_dm_dict = sigma_dm_dict
            assert all([survey in self.sigma_dm_dict.keys() for survey in self.surveys_arr]), \
                galfind_logger.critical(
                    f"All surveys in {self.surveys_arr=} must be in {self.sigma_dm_dict.keys()=}"
                )

        def _get_sigma_sq(
            self: Self,
            residuals: NDArray[float],
            params: Dict[str, float],
        ) -> NDArray[float]:
            sigma_sq = super()._get_sigma_sq(residuals, params)
            sigma_dm = np.array([self.sigma_dm_dict[survey] for survey in self.surveys_arr])
            sigma_sq += (sigma_dm * params["b_gal"] * self.y_data) ** 2
            return sigma_sq
    
    return Galaxy_Bias_Fitter