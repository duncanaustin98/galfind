"""Synthetic postage-stamp generators shared by the morphology-fitter
test suites (`test_PySersic.py`, `test_Galfit.py`), so both fitters are
exercised against the same ground truth. Not a test module itself (no
`test_*` functions) -- pytest won't collect it.
"""

import numpy as np


def make_sersic_stamp(ny=41, nx=41, amp=10.0, scale=3.0, noise=0.05, seed=0):
    """A synthetic n=1 (exponential) Sersic profile + Gaussian noise,
    with an unresolved (delta-function) PSF."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:ny, :nx]
    y0, x0 = ny / 2, nx / 2
    r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    sci = amp * np.exp(-r / scale) + rng.normal(scale=noise, size=(ny, nx))
    rms = np.full((ny, nx), noise, dtype=np.float32)
    mask = np.zeros((ny, nx), dtype=bool)
    psf = np.zeros((ny, nx), dtype=np.float32)
    psf[ny // 2, nx // 2] = 1.0
    return sci.astype(np.float32), rms, mask, psf


def make_pointsource_stamp(ny=21, nx=21, amp=10.0, noise=0.05, seed=2):
    """A synthetic unresolved point source (PSF * flux) + noise."""
    rng = np.random.default_rng(seed)
    psf = np.zeros((ny, nx), dtype=np.float32)
    psf[ny // 2, nx // 2] = 1.0
    sci = (amp * psf + rng.normal(scale=noise, size=(ny, nx))).astype(
        np.float32
    )
    rms = np.full((ny, nx), noise, dtype=np.float32)
    mask = np.zeros((ny, nx), dtype=bool)
    return sci, rms, mask, psf


def make_two_source_stamp(
    ny=51,
    nx=51,
    primary_amp=10.0,
    primary_scale=3.0,
    neighbour_offset=(12, 4),
    neighbour_amp=8.0,
    neighbour_scale=2.5,
    noise=0.05,
    seed=4,
):
    """Two overlapping synthetic n=1 Sersic sources -- a primary at the
    stamp center plus an offset neighbour -- for testing simultaneous
    (neighbour-aware) fitting. Returns `(sci, rms, mask, psf,
    primary_xy, neighbour_xy)`."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:ny, :nx]

    def sersic_blob(x0, y0, amp, scale):
        r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
        return amp * np.exp(-r / scale)

    primary_x, primary_y = nx / 2, ny / 2
    neighbour_x = primary_x + neighbour_offset[0]
    neighbour_y = primary_y + neighbour_offset[1]
    sci = (
        sersic_blob(primary_x, primary_y, primary_amp, primary_scale)
        + sersic_blob(neighbour_x, neighbour_y, neighbour_amp, neighbour_scale)
        + rng.normal(scale=noise, size=(ny, nx))
    ).astype(np.float32)
    rms = np.full((ny, nx), noise, dtype=np.float32)
    mask = np.zeros((ny, nx), dtype=bool)
    psf = np.zeros((ny, nx), dtype=np.float32)
    psf[ny // 2, nx // 2] = 1.0
    return (
        sci,
        rms,
        mask,
        psf,
        (primary_x, primary_y),
        (neighbour_x, neighbour_y),
    )
