# Multiple_Data.py

import numpy as np


class Multiple_Data:
    """Container for a collection of ``Data`` objects spanning multiple surveys.

    Parameters
    ----------
    data_arr : `list`
        Array of `Data`-like objects, one per survey, to be stored on the
        instance.

    Attributes
    ----------
    data_arr : `list`
        The stored array of `Data`-like objects.
    """

    def __init__(self, data_arr):
        self.data_arr = data_arr

    @classmethod
    def from_pipeline(
        cls, surveys, versions, instrument_names_arr, excl_bands_arr
    ):
        """Construct a `Multiple_Data` object from pipeline-style inputs.

        Normalizes the ``versions``, ``instrument_names_arr`` and
        ``excl_bands_arr`` arguments so that each has one entry per survey,
        broadcasting scalar/shared values across all surveys where needed.

        Parameters
        ----------
        surveys : `list`
            Names of the surveys to construct data for.
        versions : `str` or `list`
            Either a single pipeline version applied to all surveys, or a
            list of one version per survey.
        instrument_names_arr : `str`, `list` or `numpy.ndarray`
            Either a single instrument name, a flat list/array of instrument
            names applied to all surveys, or a nested list/array giving the
            instrument names per survey.
        excl_bands_arr : `str`, `list` or `numpy.ndarray`
            Either a single excluded band, a flat list/array of excluded
            bands applied to all surveys, or a nested list/array giving the
            excluded bands per survey.

        Returns
        -------
        `Multiple_Data`
            A new instance constructed with an (currently empty) array of
            per-survey data objects.

        Raises
        ------
        Exception
            If ``instrument_names_arr`` or ``excl_bands_arr`` is not a
            `str`, `list` or `numpy.ndarray`, or if their nested elements
            are not `str`, `list` or `numpy.ndarray`.
        """
        if type(versions) == str:
            versions = [versions for i in range(len(surveys))]
        assert len(versions) == len(surveys)

        if type(instrument_names_arr) == str:
            instrument_names_arr = [
                [instrument_names_arr] for i in range(len(surveys))
            ]
        elif type(instrument_names_arr) in [list, np.array]:
            if type(instrument_names_arr[0]) == str:
                instrument_names_arr = [
                    instrument_names_arr for i in range(len(surveys))
                ]
            elif type(instrument_names_arr[0]) in [list, np.array]:
                # already formatted correctly
                pass
            else:
                raise (Exception())
        else:
            raise (Exception())
        assert len(instrument_names_arr) == len(surveys)

        if type(excl_bands_arr) == str:
            excl_bands_arr = [[excl_bands_arr] for i in range(len(surveys))]
        elif type(excl_bands_arr) in [list, np.array]:
            if type(excl_bands_arr[0]) == str:
                excl_bands_arr = [excl_bands_arr for i in range(len(surveys))]
            elif type(excl_bands_arr[0]) in [list, np.array]:
                # already formatted correctly
                pass
            else:
                raise (Exception())
        else:
            raise (Exception())
        assert len(excl_bands_arr) == len(surveys)

        print(versions, excl_bands_arr)
        data_arr = []
        return cls(data_arr)

    @staticmethod
    def check_available_data(versions):
        """Print the available data for each pipeline version.

        Parameters
        ----------
        versions : `list`
            Pipeline versions to check for available data.

        Notes
        -----
        Not yet implemented; currently a no-op.
        """
        # print available data for each version
        pass


if __name__ == "__main__":
    Multiple_Data.from_pipeline(
        [["1"], ["2"]], "v9", "NIRCam", [["f090W", "f115W"], "f115W", "f115W"]
    )
