import h5py
import numpy as np

from galfind import config

def main():
    lam = np.array([2.198, 1.650, 1.250, 0.810, 0.650, 0.550, 0.440, 0.370, 0.296, 0.276, 0.258, 0.242, 0.229, 0.216, 0.205, 0.195, 0.186, 0.178, 0.17, 0.163, 0.157, 0.151, 0.145, 0.14, 0.136, 0.131, 0.127, 0.123, 0.119, 0.116])
    Alam_AV = np.array([0.016, 0.169, 0.131, 0.567, 0.801, 1.0, 1.374, 1.672, 2.0, 2.22, 2.428, 2.661, 2.947, 3.161, 3.293, 3.489, 3.637, 3.866, 4.013, 4.243, 4.472, 4.776, 5.0, 5.272, 5.575, 5.795, 6.074, 6.297, 6.436, 6.992])
    Alam_AV_err = np.array([0.003, 0.02, 0.013, 0.048, 0.113, 0.046, 0.127, 0.123, 0.095, 0.093, 0.093, 0.095, 0.099, 0.102, 0.104, 0.105, 0.107, 0.112, 0.115, 0.119, 0.124, 0.131, 0.135, 0.142, 0.148, 0.153, 0.16, 0.368, 0.271, 0.201])
    # Save the data to an HDF5 file
    with h5py.File(f"{config['DEFAULT']['GALFIND_DIR']}/../data/SMC_Gordon+03.h5", "w") as f:
        f.create_dataset("lam", data=lam)
        f.create_dataset("Alam_AV", data=Alam_AV)
        f.create_dataset("Alam_AV_err", data=Alam_AV_err)
        f.close()

if __name__ == "__main__":
    main()