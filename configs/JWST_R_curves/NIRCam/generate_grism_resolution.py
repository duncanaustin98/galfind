import numpy as np

plot = True

def grism_resolution(wavelength):
    wavelength =  np.array(wavelength)
    return 3.35 * wavelength**4 - 41.9 * wavelength**3 + 95.5 * wavelength**2 + 536 * wavelength - 240
# in Angstrom
wavelengths = np.arange(2.5, 5.2, 0.0001) 

resolutions = grism_resolution(wavelengths)

#get file directory
import os
current_dir = os.path.dirname(os.path.realpath(__file__))


if plot:
    import matplotlib.pyplot as plt
    plt.plot(wavelengths, resolutions)
    plt.xlabel('Wavelength (µm)')
    plt.ylabel('Resolution')
    plt.title('NIRCam Grism Resolution')
    plt.savefig(f'{current_dir}/nircam_grism_resolution.png')

# Save to csv

data = np.array([wavelengths, resolutions]).T

np.savetxt(f'{current_dir}/nircam_grism_resolution.csv', data, delimiter=',', header='Wavelength (um), R', comments='')
