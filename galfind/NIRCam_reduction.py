from __future__ import annotations

try:
    from typing import Type  # python 3.11+
except ImportError:
    from typing_extensions import Type  # python > 3.7 AND python < 3.11
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import Instrument
from astroquery.mast import Observations

from .decorators import run_in_dir
from . import config

@run_in_dir(f"{config['DEFAULT']['GALFIND_DATA']}/test")
def query_mast(instrument: Type[Instrument], proposal_id: int) -> Observations:
    instrument_name = f"{instrument.__class__.__name__.upper()}/IMAGE"
    obs_table = Observations.query_criteria(instrument_name=instrument_name, proposal_id=str(proposal_id))
    print(obs_table, obs_table["target_name"], obs_table.colnames)
    data_products = Observations.get_product_list(obs_table)
    #manifest = Observations.download_products(data_products, productSubGroupDescription='UNCAL', curl_flag=True)