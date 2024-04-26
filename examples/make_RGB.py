# make_RGB.py

from galfind import Data


def make_RGB(survey, version, blue_bands = ["F090W"], green_bands = ["F200W"], \
        red_bands = ["F444W"], instruments = ["NIRCam"], excl_bands = [], RGB_method = "trilogy"):
    data = Data.from_pipeline(survey, version, instruments, excl_bands)
    # make RGB using trilogy by default
    data.make_RGB(blue_bands = blue_bands, green_bands = green_bands, red_bands = red_bands, method = RGB_method)

if __name__ == "__main__":
    survey = "JOF"
    version = "v11"
    make_RGB(survey, version)