#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 13:54:02 2023

@author: u92876da
"""

import astropy.units as u
import matplotlib.pyplot as plt

from galfind import Filter, Multiple_Filter


def main():
    miri = Multiple_Filter.from_instrument("MIRI", keep_suffix = "W") - "F1130W"
    fig, ax = plt.subplots()
    miri.plot(ax, save=True, show=False)


if __name__ == "__main__":
    main()
