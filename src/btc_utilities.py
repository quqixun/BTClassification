# Brain Tumor Classification
# Script for Utility Functions
# Author: Qixun Qu
# Create on: 2017/10/11
# Modify on: 2017/10/11

#     ,,,         ,,,
#   ;"   ';     ;'   ",
#   ;  @.ss$$$$$$s.@  ;
#   `s$$$$$$$$$$$$$$$'
#   $$$$$$$$$$$$$$$$$$
#  $$$$P""Y$$$Y""W$$$$$
#  $$$$  p"$$$"q  $$$$$
#  $$$$  .$$$$$.  $$$$'
#   $$$DaU$$O$$DaU$$$'
#    '$$$$'.^.'$$$$'
#       '&$$$$$&'


import numpy as np
from math import factorial


def compute_hist(volume):
    rvolume = np.round(volume)
    bins = np.arange(np.min(rvolume), np.max(rvolume))
    hist = np.histogram(rvolume, bins=bins, density=True)

    x = hist[1][1:]
    y = hist[0]

    i = np.where(y > 0)
    bg_index = i[0][1] + int((i[0][-1] - i[0][1]) / 20.)
    bg = x[bg_index]

    x = x[bg_index:]
    y = y[bg_index:]

    ysg = savitzky_golay(y, window_size=31, order=4)

    return x, y, ysg, bg


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order + 1)
    half_window = (window_size - 1) // 2

    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)

    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])

    y = np.concatenate((firstvals, y, lastvals))

    return np.convolve(m[::-1], y, mode='valid')
