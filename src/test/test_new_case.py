import os
import numpy as np
import pandas as pd
from math import factorial
from btc_settings import *
import matplotlib.pyplot as plt


def plot_volumes(volume, temp, title):
    for i in range(shape[2]):
        plt.figure()
        plt.subplot(2, 4, 1)
        plt.axis("off")
        plt.title(title)
        plt.imshow(volume[:, :, i, 0], cmap="gray")
        plt.subplot(2, 4, 2)
        plt.axis("off")
        plt.imshow(temp[:, :, i, 0], cmap="gray")
        plt.subplot(2, 4, 3)
        plt.axis("off")
        plt.imshow(volume[:, :, i, 1], cmap="gray")
        plt.subplot(2, 4, 4)
        plt.axis("off")
        plt.imshow(temp[:, :, i, 1], cmap="gray")
        plt.subplot(2, 4, 5)
        plt.axis("off")
        plt.imshow(volume[:, :, i, 2], cmap="gray")
        plt.subplot(2, 4, 6)
        plt.axis("off")
        plt.imshow(temp[:, :, i, 2], cmap="gray")
        plt.subplot(2, 4, 7)
        plt.axis("off")
        plt.imshow(volume[:, :, i, 3], cmap="gray")
        plt.subplot(2, 4, 8)
        plt.axis("off")
        plt.imshow(temp[:, :, i, 3], cmap="gray")
        plt.show()


def plot_hist(hist):
    x = hist[1][1:]
    y = hist[0]
    i = np.where(y > 0)
    x = x[i]
    y = y[i]
    plt.plot(x, y)
    plt.show()


# Smooth the histogram curve
# by Savitzky-Golay filter
# http://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
# https://www.wire.tu-bs.de/OLDWEB/mameyer/cmr/savgol.pdf
# https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

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
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def plot_filtered_curve(x, y1, y2):
    plt.plot(x, y1, label='Original signal')
    plt.plot(x, y2, 'r', label='Filtered signal')
    plt.legend()
    plt.show()


def plot_two_curves(x1, y1, label1, x2, y2, label2):
    plt.plot(x1, y1, label=label1)
    plt.plot(x2, y2, label=label2)
    plt.legend()
    plt.show()


def compute_hist(arr):
    arrr = np.round(arr)
    hist = np.histogram(arrr, bins=np.arange(np.min(arrr), np.max(arrr)), density=True)
    x = hist[1][1:]
    y = hist[0]
    i = np.where(y > 0)
    # if not background:
    bg_index = i[0][1] + int((i[0][-1] - i[0][1]) / 20.)
    bg = x[bg_index]
    x = x[bg_index:]
    y = y[bg_index:]
    # else:
    # x = x[i[0][1]:]
    # y = y[i[0][1]:]
    ysg = savitzky_golay(y, window_size=31, order=4)
    return x, y, ysg, bg


parent_dir = os.path.dirname(os.getcwd())
labels = pd.read_csv(os.path.join(parent_dir, "data", "label.csv"))

grade_ii_index = np.where(labels["Grade_Label"] == GRADE_II_LABEL)[0]
grade_ii_cases = labels["Case"][grade_ii_index].values.tolist()

idx = 10
chn = 1
folder = "Temp/resize"
file = os.path.join(folder, grade_ii_cases[idx] + ".npy")
volume = np.load(file)
shape = volume.shape
temp = np.copy(volume)

for i in range(shape[2]):
    temp[:, :, i, :] = np.fliplr(temp[:, :, i, :])
# plot_volumes(volume, temp, title=grade_ii_cases[idx])

temp_chn = temp[..., chn]
x, y, ysg, bg = compute_hist(temp_chn)
# plot_filtered_curve(x, y, ysg)

# temp_chn_mean = np.mean(temp[..., chn])
# temp_mean_diff = temp_chn_mean * 0.2
# temp_mean_sign = np.random.randint(2, size=1)[0] * 2 - 1
# new_temp_chn = temp_chn + int(temp_mean_sign * temp_mean_diff)
# new_temp_chn[new_temp_chn < 0] = 0
# nx, ny, nysg, nbg = compute_hist(new_temp_chn)
# plot_two_curves(x, ysg, "Original", nx, nysg, "Change Mean")


# Modify Intensity
new_temp_chn = temp_chn
print("Original mean: {0}, original std: {1}".format(np.mean(temp_chn), np.std(temp_chn)))


new_temp_chn = np.reshape(new_temp_chn, ((1, -1)))[0]
non_bg_index = np.where(new_temp_chn > 0)  # nbg
for i in non_bg_index:
    sign = np.random.randint(2, size=1)[0] * 2 - 1
    scope = np.random.randint(1, 11, size=1)[0] / 100
    new_temp_chn[i] = new_temp_chn[i] * (1 + sign * scope)
new_temp_chn = np.reshape(new_temp_chn, ((59, 59, 59)))
mx, my, mysg, mbg = compute_hist(new_temp_chn)
print("Modified mean: {0}, modified std: {1}".format(np.mean(new_temp_chn), np.std(new_temp_chn)))
plot_two_curves(x, ysg, "Original", mx, mysg, "Change Intensity")


# non_bg_index = np.where(new_temp_chn > bg)  # nbg
# new_temp_chn_sort = np.sort(new_temp_chn[non_bg_index])
# new_sort_len = len(new_temp_chn_sort)


# pcts_len = 11
# half_pcts_len = int((pcts_len - 1) / 2)
# pcts = np.linspace(0.0, 1.0, pcts_len, dtype=np.float32)
# original_pcts = []
# for p in pcts:
    # pct_index = int(np.ceil(new_sort_len * p))
    # if pct_index == new_sort_len:
        # pct_index -= 1
    # original_pcts.append(new_temp_chn_sort[pct_index])
# original_pcts = np.array(original_pcts)
# modified_pcts = []
# temp_std_sign = np.random.randint(2, size=1)[0] * 2 - 1

# modified_pcts += [p * (1 + temp_std_sign * 0.001) for p in original_pcts[:half_pcts_len]]
# modified_pcts += [p * (1 - temp_std_sign * 0.001) for p in original_pcts[half_pcts_len:]]

# new_temp_chn[non_bg_index] = np.interp(new_temp_chn[non_bg_index], original_pcts, modified_pcts)
# print(np.std(new_temp_chn))
# mx, my, mysg, mbg = compute_hist(new_temp_chn)
# plot_two_curves(nx, nysg, "Change Mean", mx, mysg, "Change Mean & Std")
# plot_two_curves(x, ysg, "Original", mx, mysg, "Change Mean & Std")


# vmin = 0
# vmax = np.max([np.max(temp_chn), np.max(new_temp_chn)])
# for i in range(59):
#     plt.subplot(1, 2, 1)
#     plt.imshow(volume[:, :, i, chn], cmap="gray", vmin=vmin, vmax=vmax)
#     plt.subplot(1, 2, 2)
#     plt.imshow(new_temp_chn[:, :, i], cmap="gray", vmin=vmin, vmax=vmax)
#     plt.show()
