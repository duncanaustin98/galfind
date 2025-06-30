
from __future__ import annotations

from abc import ABC, abstractmethod
import emcee
import numpy as np
import corner
import multiprocessing as mp
import os
import json
from astropy.stats import sigma_clip
from scipy import stats
from tqdm import tqdm
import h5py
from matplotlib import patheffects as pe
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import NoReturn, Union, Optional, List, Dict, Any, Tuple, TYPE_CHECKING
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11

from . import config, galfind_logger

class Prior(ABC):

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

    def __call__(
        self: Self,
        param: float
    ) -> bool:
        if param > self.prior_params["lower_lim"] and \
                param < self.prior_params["upper_lim"]:
            return True
        else:
            return False
        

class Priors:

    def __init__(
        self: Self,
        prior_arr: List[Prior],
    ):
        self.prior_arr = prior_arr
    
    @property
    def names(self):
        return [prior.name for prior in self.prior_arr]

    def __call__(
        self: Self,
        params: Dict[str, float]
    ) -> float:
        if all(prior(params[prior.name]) for prior in self.prior_arr):
            return 0.0
        else:
            return -np.inf

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
                backend_filename = f"{backend_filename}.h5"
            self.backend = emcee.backends.HDFBackend(backend_filename)
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_likelihood, backend = self.backend) #, blobs_dtype = blobs_dtype, pool = pool)
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
        return len(self.priors)
    
    @property
    def fiducial_params(self):
        return [prior.fiducial for prior in self.priors]
    
    def _instantiate_walkers(self: Self) -> NoReturn:
        # # uniformly distributed starting positions over flat prior
        # flat_priors_lower = [self.flat_priors[i][0] for i in range(self.ndim)]
        # flat_priors_diff = [self.flat_priors[i][1] - self.flat_priors[i][0] for i in range(self.ndim)]

        # pos = [flat_priors_lower + (flat_priors_diff) \
        #        * np.random.uniform(0, 1, self.ndim) for i in range(self.nwalkers)]
        if not hasattr(self, "init_pos"):
            init_pos = [self.fiducial_params + 1e-4 * np.random.uniform(0, 1, self.ndim) * \
                    self.fiducial_params for i in range(self.nwalkers)]
            # init_pos = [np.array([np.random.uniform(prior.prior_params["lower_lim"], \
            #     prior.prior_params["upper_lim"], 1)[0] for prior in self.priors]) \
            #     for i in range(self.nwalkers)]
            self.init_pos = init_pos

    def __call__(
        self: Self,
        n_steps: int,
        n_processes: int = 1,
    ) -> NoReturn:
        if hasattr(self, "backend"):
            galfind_logger.info("Initial size: {0}".format(self.backend.iteration))
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
        pass
    
    @abstractmethod
    def model(
        self: Self,
        x: Union[float, List[float], NDArray[float]],
        params: Dict[str, float]
    ) -> float:
        pass
    
    # @abstractmethod
    # def blob_funcs(self):
    #     pass
    
    @abstractmethod
    def log_likelihood(
        self: Self,
        params: List[float],
    ) -> float:
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
        label: Optional[str] = None,
        colour: str = "black",
        **kwargs: Dict[str, Any]
    ) -> None: #nsamples = 1_000, plot_med = False, alpha = 0.3):
        if x_arr is None:
            x_arr = np.linspace(np.min(self.x_data), np.max(self.x_data), 100)

        l1_chains, med_chains, u1_chains = self._get_plot_chains(x_arr = x_arr, log_data = log_data)
        
        ax.plot(x_arr, med_chains, color = colour, zorder = 200, label = label)
        ax.fill_between(x_arr, l1_chains, u1_chains, color = colour, alpha = 0.5, \
            zorder = 200, path_effects = [pe.withStroke(linewidth = 2., foreground = "white")])
        galfind_logger.info("Plotting MCMC fit")

    def get_sample(self: Self) -> NDArray[float]:
        autocorr_time = np.max(self.sampler.get_autocorr_time())
        discard = int(autocorr_time * 2)
        thin = int(autocorr_time / 2)
        chain = self.sampler.get_chain(flat = True, discard = discard, thin = thin)
        return chain
    
    def _get_plot_chains(
        self: Self,
        x_arr: NDArray[float],
        log_data: bool = False,
    ) -> Tuple[NDArray[float], NDArray[float], NDArray[float]]:
        y_fit = self.get_chains(x_arr, log_data = log_data).T
        l1_chains = np.array([np.percentile(y, 16) for y in y_fit])
        med_chains = np.array([np.percentile(y, 50) for y in y_fit])
        u1_chains = np.array([np.percentile(y, 84) for y in y_fit])
        return l1_chains, med_chains, u1_chains
    
    def get_chains(
        self: Self,
        x_arr: NDArray[float],
        log_data: bool = False,
        shape: Optional[int] = None,
    ) -> NDArray[float]:
        autocorr_time = np.max(self.sampler.get_autocorr_time())
        discard = int(autocorr_time * 2)
        thin = int(autocorr_time / 2)
        if log_data:
            model = lambda x, params: np.log10(self.model(x, params))
        else:
            model = self.model
        if shape is None:
            params_arr = self.sampler.get_chain(flat = True, discard = discard, thin = thin)
        else:
            params_arr = self.sampler.get_chain(flat = True, discard = discard, thin = thin)[-shape:]
        y_fit = np.array([model(x_arr, {prior.name: param for prior, param \
            in zip(self.priors, params)}) for params in tqdm( \
            params_arr, desc = "Loading chains", total = len(params_arr))])
        return y_fit

    def plot_corner(
        self: Self,
        fig: Optional[plt.Figure] = None,
        range: Optional[List[Tuple[float]]] = None,
        colour: str = "black",
        legend: bool = False,
        save: bool = True,
        **plot_kwargs: Dict[str, Any]
    ) -> plt.Figure:

        flat_samples = self.backend.get_chain(flat = True) #, discard=100, thin=15)

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

        if save == True:
            fig_.savefig(self.backend_filename.replace(".h5", "_MCMC.jpeg"), dpi = 600, bbox_inches = "tight")
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
        pass
    
    @abstractmethod
    def model(
        self: Self,
        x: Union[float, List[float], NDArray[float]],
        params: Dict[str, float]
    ) -> float:
        pass

    def _get_sigma_sq(
        self: Self,
        residuals: NDArray[float],
        params: Dict[str, float],
    ) -> NDArray[float]:
        sigma = [y_err_u1 if residual > 0. else y_err_l1 for residual, y_err_u1, y_err_l1 in 
            zip(residuals, self.y_data_errs[0], self.y_data_errs[1])]
        return np.array(sigma) ** 2
    
    def log_likelihood(
        self: Self,
        params: List[float],
    ) -> float:
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
    
    def get_scatter(self) -> float:
        residuals = self.get_residuals(self.get_params_med())
        self.scatter = np.sqrt(np.sum(residuals ** 2) / (len(self.y_data) - 1))
        galfind_logger.info(f"Scatter: {self.scatter:.3f} dex")
        return self.scatter

    # def get_sample(self: Self) -> NDArray[float]:
    #     autocorr_time = np.max(self.sampler.get_autocorr_time())
    #     discard = int(autocorr_time * 2)
    #     thin = int(autocorr_time / 2)
    #     chain = self.sampler.get_chain(flat = True, discard = discard, thin = thin)
    #     return chain

    def get_params_med(self: Self) -> Dict[str, float]:
        if not hasattr(self, "params_med"):
            chain = self.get_sample()
            self.params_med = {prior.name: np.median(chain[:, i]) for i, prior in enumerate(self.priors)}
        galfind_logger.info(f"Median parameters: {self.params_med}")
        return self.params_med
    
    def get_params_errs(self: Self) -> NDArray[float]:
        if not hasattr(self, "params_errs"):
            chain = self.get_sample()
            params = {prior.name: np.percentile(chain[:, i], [16, 50, 84]) for i, prior in enumerate(self.priors)}
            self.params_errs = {}
            for key, vals in params.items():
                self.params_errs[key] = np.array([vals[1] - vals[0], vals[2] - vals[1]])
        galfind_logger.info(f"{self.params_errs}=")
        return self.params_errs
    
    def get_residuals(self: Self, params: Dict[str, float]) -> NDArray[float]:
        return self.y_data - self.model(self.x_data, params)

    def sigma_clip(
        self: Self,
        sigma: float = 3.0
    ) -> NDArray[bool]:
        removed_residuals = sigma_clip(self.get_residuals(self.get_params_med()), sigma = sigma, masked = True)
        kept_residuals = ~removed_residuals.mask
        return kept_residuals

class Scattered_MCMC_Fitter(Base_MCMC_Fitter):

    def __init__(
        self: Self,
        priors: Priors,
        x_data_chains: NDArray[NDArray[float]],
        y_data_chains: NDArray[NDArray[float]],
        nwalkers: int,
        backend_filename: Optional[str],
        fixed_params: Dict[str, float],
    ):
        assert "scatter" in priors.names + list(fixed_params.keys())
        y_data_errs = None
        super().__init__(priors, x_data_chains, y_data_chains, y_data_errs, nwalkers, backend_filename, fixed_params)
        self.mask = self.x_data.mask | self.y_data.mask

    @abstractmethod
    def update(self):
        pass
    
    @abstractmethod
    def model(
        self: Self,
        x: Union[float, List[float], NDArray[float]],
        params: Dict[str, float]
    ) -> float:
        pass

    def _get_plot_chains(
        self: Self,
        x_arr: NDArray[float],
        log_data: bool = False,
    ) -> Tuple[NDArray[float], NDArray[float], NDArray[float]]:
        autocorr_time = np.max(self.sampler.get_autocorr_time())
        discard = int(autocorr_time * 2)
        thin = 1 # int(autocorr_time / 2)
        if log_data:
            model = lambda x, params: np.log10(self.model(x, params))
        else:
            model = self.model
        params_arr = [{prior.name: param for prior, param in zip(self.priors, params)} \
            for params in self.sampler.get_chain(flat = True, discard = discard, thin = thin)]
        y_fit = np.array([model(x_arr, params) for params in params_arr]).T
        l1_chains = np.array([np.percentile(y, 16) for y in y_fit])
        med_chains = np.array([np.percentile(y, 50) for y in y_fit])
        u1_chains = np.array([np.percentile(y, 84) for y in y_fit])
        return l1_chains, med_chains, u1_chains

    def log_likelihood(
        self: Self,
        params: List[float],
    ) -> float:
        params_loc = {}
        for i, prior in enumerate(self.priors):
            params_loc[prior.name] = params[i]
        lp = self.priors(params_loc)
        if not np.isfinite(lp):
            return -np.inf # np.full(1 + len(self.blob_keys), -np.inf)
        else:
            model = self.model(self.x_data, params_loc)
            model.mask = self.mask
            probs_samples = stats.norm(model, params_loc["scatter"]).pdf(self.y_data)
            probs_objects = np.nanmean(probs_samples, axis = 1)
            assert len(probs_objects) == len(self.x_data)
            return np.log(probs_objects + 1e-100).sum()


class Schechter_Lum_Fitter(MCMC_Fitter):

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
        params: Dict[str, float]
    ) -> float:
        return 10 ** params["log10_phi_star"] * ((x / params["L_star"]) ** params["alpha"]) * \
            np.exp(-x / params["L_star"]) / params["L_star"]
    
    def update(
        self: Self
    ) -> NoReturn:
        pass


class Schechter_Mag_Fitter(MCMC_Fitter):
    
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
        super().__init__(priors, x_data, y_data, y_data_errs, nwalkers, backend_filename, fixed_params)
    
    def model(
        self: Self,
        x: Union[float, List[float], NDArray[float]],
        params: Dict[str, float]
    ) -> float:
        return (10 ** params["log10_phi_star"]) * 0.4 * np.log(10) * \
            10 ** (0.4 * (params["alpha"] + 1) * (params["M_star"] - x)) * \
            np.exp(-10 ** (0.4 * (params["M_star"] - x)))
    
    def update(
        self: Self
    ) -> NoReturn:
        pass

class Linear_Fitter(MCMC_Fitter):
    # Power Law in linear space
    def __init__(
        self: Self,
        priors: Priors,
        x_data: NDArray[float],
        y_data: NDArray[float],
        y_data_errs: NDArray[NDArray[float, float]],
        nwalkers: int,
        backend_filename: Optional[str],
        fixed_params: Dict[str, float],
        incl_scatter: bool = False,
    ):
        self.incl_scatter = incl_scatter
        params = ["m", "c"]
        if incl_scatter:
            params.append("scatter")
        assert all([key in params for key in fixed_params.keys()]), \
            galfind_logger.critical(
                f"{fixed_params=} must be m and/or c or empty"
            )
        assert len(fixed_params) + len(priors) == len(params), \
            galfind_logger.critical(
                f"Must have exactly {len(params)} parameters if " + \
                f"{'not' if not incl_scatter else ''} including scatter"
            )
        assert all([key in fixed_params.keys() or key in priors.names for key in params]), \
            galfind_logger.critical(
                f"{repr(priors)=} or {fixed_params=} must contain {', '.join(params)}"
            )
        super().__init__(priors, x_data, y_data, y_data_errs, nwalkers, backend_filename, fixed_params)
    
    @classmethod
    def from_h5(cls: Type[Self], h5_path: str) -> Self:
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
        return cls(priors, x_data, y_data, y_data_errs, n_walkers, backend_filename = backend_filename, fixed_params = {})#, init_pos = init_pos)
    
    def save_h5(self: Self) -> NoReturn:
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
        out_file.create_dataset("incl_scatter", data = self.incl_scatter)
        out_file.create_dataset("backend_filename", data = self.backend_filename)
        out_file.close()

    def _get_sigma_sq(
        self: Self,
        residuals: NDArray[float],
        params: Dict[str, float],
    ) -> NDArray[float]:
        if self.incl_scatter:
            return super()._get_sigma_sq(residuals, params) + params["scatter"] ** 2 #\
                #np.exp(2 * params["logf"]) * self.model(self.x_data, params) ** 2
        else:
            return super()._get_sigma_sq(residuals, params)

    def model(
        self: Self,
        x: Union[float, List[float], NDArray[float]],
        params: Dict[str, float]
    ) -> float:
        return params["m"] * x + params["c"]
    
    def update(
        self: Self
    ) -> NoReturn:
        pass

class Power_Law_Fitter(MCMC_Fitter):
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
        params: Dict[str, float]
    ) -> float:
        return params["A"] * x ** params["slope"]
    
    def update(
        self: Self
    ) -> NoReturn:
        pass

    # def plot(
    #     self: Self,
    #     ax: plt.Axes,
    #     log_data: bool = True,
    #     x_arr: Optional[NDArray[float]] = None,
    #     **kwargs: Dict[str, Any]
    # ) -> None:
    #     super().plot(ax, log_data, x_arr, **kwargs)

# below is copied!
class Scattered_Linear_Fitter(Scattered_MCMC_Fitter):

    def __init__(
        self: Self,
        priors: Priors,
        x_data_chains: NDArray[NDArray[float]],
        y_data_chains: NDArray[NDArray[float]],
        nwalkers: int,
        backend_filename: Optional[str],
        fixed_params: Dict[str, float],
        incl_logf: bool = False,
    ):
        self.incl_logf = incl_logf
        params = ["m", "c", "scatter"]
        if incl_logf:
            params.append("logf")
        assert all([key in params for key in fixed_params.keys()]), \
            galfind_logger.critical(
                f"{fixed_params=} must be m and/or c or empty"
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
        super().__init__(priors, x_data_chains, y_data_chains, nwalkers, backend_filename, fixed_params)

    def model(
        self: Self,
        x: Union[float, List[float], NDArray[float]],
        params: Dict[str, float]
    ) -> float:
        return params["m"] * x + params["c"]
    
    def update(
        self: Self
    ) -> NoReturn:
        pass

