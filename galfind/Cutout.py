
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import astropy.units as u
from copy import deepcopy
from astropy.coordinates import SkyCoord
from typing import List, Union, NoReturn, TYPE_CHECKING
if TYPE_CHECKING:
    from . import Data
    from . import Galaxy
    from . import Catalogue
    from . import Filter
    from . import Multiple_Filter
try:
    from typing import Self, Type  # python 3.11+
except ImportError:
    from typing_extensions import Self, Type  # python > 3.7 AND python < 3.11


class Cutout_Base(ABC):

    def __init__(self, cutout_path: str, filt: Filter, cutout_size: u.Quantity) -> Self:
        self.cutout_path = cutout_path
        self.filt = filt
        self.cutout_size = cutout_size

    def __copy__(self) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result

    def load(self) -> np.ndarray:
        pass

    def plot(self, ax) -> NoReturn:
        cutout_data = self.load()
        ax.imshow(cutout_data)

class Cutout(Cutout_Base):

    @classmethod
    def from_gal(cls: Type[Self], data: Data, filt: Filter, gal: Galaxy, cutout_size: u.Quantity) -> Self:
        # extract the position of the galaxy
        sky_coord = gal.sky_coord
        # save path is the source ID + survey + version + band of the galaxy
        save_path = ""
        return cls.from_data_skycoord(data, filt, sky_coord, cutout_size, save_path)

    @classmethod
    def from_data_skycoord(cls: Type[Self], data: Data, filt: Filter, sky_coord: SkyCoord, cutout_size: u.Quantity, save_path: Union[str, None] = None) -> Self:
        if save_path is None:
            # save path is the RA/DEC + survey + version + band of the galaxy
            save_path = ""
        # make cutout from data at the sky co-ordinate and save
        cls._make_cutout(data, sky_coord, cutout_size, save_path)
        return cls(save_path, filt, cutout_size)

    @staticmethod
    def _make_cutout(data: Data, sky_coord: SkyCoord, cutout_size: u.Quantity, save_path: str) -> NoReturn:
        # make cutout from data at the sky co-ordinate
        pass

    def __add__(self, other: Union[Cutout, List[Cutout]]) -> Union[Stacked_Cutout, RGB]:
        # make other a list of Cutout objects if not already
        if isinstance(other, Cutout):
            other = [other]
        # stack cutouts that are from the same filter

        # make an RGB if all
        # ensure all cutout filters are the same
        if not all([cutout.filter == self.filter for cutout in other]):
            raise ValueError(f"All cutouts must have the same filter as {repr(self.filter)=}")


class Stacked_Cutout(Cutout_Base):
    
    def __init__(self, cutout_path: str, filt: Filter, cutout_size: u.Quantity, origin_paths: List[str], ) -> Self:
        self.origin_paths = origin_paths
        super().__init__(cutout_path, filt, cutout_size)

    @classmethod
    def from_cat(cls, cat: Catalogue, filt: Filter, cutout_size: u.Quantity, save_path: str = None) -> Self:
        # make every individual cutout from the catalogue
        cutouts = [Cutout.from_gal(cat.data, filt, gal, cutout_size) for gal in cat]
        return cls.from_cutouts(cutouts, save_path)
    
    @classmethod
    def from_data_skycoords(cls, data: Data, filt: Filter, sky_coords: Union[SkyCoord, List[SkyCoord]], cutout_size: u.Quantity, save_path: str = None) -> Self:
        # make every individual cutout from the data at the given SkyCoord
        cutouts = [Cutout.from_data_skycoord(data, filt, sky_coord, cutout_size, save_path) for sky_coord in sky_coords]
        return cls.from_cutouts(cutouts, save_path)
    
    @classmethod
    def from_cutouts(cls, cutouts: List[Cutout], save_path: str = None) -> Self:
        # ensure all cutouts are from the same filter
        assert all([cutout.filter == cutouts[0].filter for cutout in cutouts])
        # stack cutouts if they have not been already
        cls._stack_cutouts(cutouts, save_path)
        # extract original cutout paths
        origin_paths = [cutout.cutout_path for cutout in cutouts]
        return cls(save_path, origin_paths, cutouts[0].filter)
    
    @staticmethod
    def _stack_cutouts(cutouts: List[Cutout], save_path: str) -> NoReturn:
        """
        Stack cutouts to create a stacked cutout
        """
        pass


class RGB_Base(ABC):
    
    def __init__(self, cutouts: List[Cutout, Stacked_Cutout]) -> Self:
        self.cutouts = cutouts
    
    def __len__(self) -> int:
        return len(self.cutouts)
    
    def __iter__(self):
        return iter(self.cutouts)
    
    def __next__(self):
        if self.iter > len(self) - 1:
            raise StopIteration
        else:
            cutout = self[self.iter]
            self.iter += 1
            return cutout
        
    def __getitem__(self, index: int) -> Type[Self]:
        # improve here
        return self.cutouts[index]

    def __copy__(self) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, deepcopy(value, memo))
        return result
    
    # need to determine whether this is a good place for this
    def __add__(self):
        pass

    @property
    def filterset(self) -> Multiple_Filter:
        return Multiple_Filter([deepcopy(cutout.filter) for cutout in self])

    def load(self, band) -> np.ndarray:
        return self[band].load()

    def plot(self):
        pass


class RGB(RGB_Base):

    def __init__(self, cutouts: List[Cutout]) -> Self:
        # ensure all cutouts are from different filters
        assert all([cutout.filt != cutouts[0].filt for cutout in cutouts])
        super().__init__(cutouts)

    @classmethod
    def from_gal(cls: Type[Self], data: Data, gal: Galaxy) -> Self:
        # make a cutout for each filter
        cutouts = [Cutout.from_gal(data, filt, gal) for filt in data.filterset]
        return cls(cutouts)
    
    @classmethod
    def from_data_skycoord(cls: Type[Self], data: Data, sky_coord: SkyCoord) -> Self:
        # make a cutout for each filter
        cutouts = [Cutout.from_data_skycoord(data, filt, sky_coord) for filt in data.filterset]
        return cls(cutouts)
    

class Stacked_RGB(RGB_Base):

    def __init__(self, stacked_cutouts: List[Stacked_Cutout]) -> Self:
        # ensure all stacked cutouts are from different filters
        assert all(all([cutout.filt != cutouts[0].filt for cutout in cutouts] for cutouts in stacked_cutouts))
        super().__init__(stacked_cutouts)

    @classmethod
    def from_cat(cls: Type[Self], cat: Catalogue) -> Self:
        # make a stacked cutout for each filter
        stacked_cutouts = [Stacked_Cutout.from_cat(cat, filt) for filt in cat.data.filterset]
        return cls(stacked_cutouts)
    
    @classmethod
    def from_data_skycoords(cls: Type[Self], data: Data, sky_coords: Union[SkyCoord, List[SkyCoord]]) -> Self:
        # make a stacked cutout for each filter
        stacked_cutouts = [Stacked_Cutout.from_data_skycoords(data, filt, sky_coords) for filt in data.filterset]
        return cls(stacked_cutouts)

class Multiple_Cutout_Base(ABC):
    pass

class Multiple_Cutout(Multiple_Cutout_Base):
    pass

class Multiple_RGB(Multiple_Cutout_Base):
    pass