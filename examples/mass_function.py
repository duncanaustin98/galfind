# mass_function.py
from astropy.table import Table

from galfind import Base_Number_Density_Function, Number_Density_Function

def conv_tharvey_flags():
    # open .ecsv
    tab = Table.read("/nvme/scratch/work/austind/flags_data/flags_data/data/DistributionFunctions/Mstar/obs/binned/harvey24.ecsv")
    #tab.rename_column("phi_error_low", "phi_err_low")
    #tab.rename_column("phi_error_upp", "phi_err_upp")
    tab.meta = {**tab.meta, **{"redshifts": [7., 8., 9., 10.5, 12.5]}}
    tab.write("/nvme/scratch/work/austind/flags_data/flags_data/data/DistributionFunctions/Mstar/obs/binned/harvey24.ecsv", overwrite = True)
    breakpoint()

def main():
    test_flags_func = Base_Number_Density_Function.from_flags_repo("stellar_mass", [8.5, 9.5], "Harvey+24")

if __name__ == "__main__":
    main()
    #conv_tharvey_flags()