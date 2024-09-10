import pytest

from galfind import Instrument


# TODO: This currently won't work as Filter.__add__ needs finishing
@pytest.fixture(scope="session")
def PEARLS_instr():
    incl_bands = [
        "F090W",
        "F115W",
        "F150W",
        "F200W",
        "F277W",
        "F356W",
        "F410M",
        "F444W",
    ]
    return Instrument.from_bands(incl_bands)
