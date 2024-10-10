from galfind import Instrument, NIRCam
from galfind.NIRCam_reduction import query_mast

query_mast(instrument=NIRCam(), proposal_id=2282)