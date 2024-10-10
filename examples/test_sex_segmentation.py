
from galfind import Data
from galfind.Data import morgan_version_to_dir
from galfind import config

def main():
    JOF_data = Data.from_survey_version("JOF", "v11", instrument_names = ["NIRCam"], version_to_dir_dict = morgan_version_to_dir)
    F444W_JOF = JOF_data["F444W"]
    print(F444W_JOF)
    F444W_JOF.segment()

main()