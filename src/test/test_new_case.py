import os
import numpy as np
import pandas as pd
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


parent_dir = os.path.dirname(os.getcwd())
labels = pd.read_csv(os.path.join(parent_dir, "data", "label.csv"))

grade_ii_index = np.where(labels["Grade_Label"] == GRADE_II_LABEL)[0]
grade_ii_cases = labels["Case"][grade_ii_index].values.tolist()


idx = 2
chn = 3
folder = "Temp/resize"
file = os.path.join(folder, grade_ii_cases[idx] + ".npy")
volume = np.load(file)
shape = volume.shape
temp = np.copy(volume)

for i in range(shape[2]):
    temp[:, :, i, :] = np.fliplr(temp[:, :, i, :])

# plot_volumes(volume, temp, title=grade_ii_cases[idx])
# print(np.min(temp), np.max(temp))
temp = np.round(temp)
# print(np.min(temp), np.max(temp))
hist = np.histogram(temp[..., chn], bins=np.arange(1, np.max(temp[..., chn])), density=True)

# print(len(hist[0]), len(hist[1]))

x = hist[1][1:]
y = hist[0]
i = np.where(y > 0)
# print(len(i[0]))
# print(i[0][0])
# bg = i[0][0] + 20
bg = i[0][0] + int((i[0][-1] - i[0][0]) / 20)
x = x[bg:]
y = y[bg:]


# plot_hist(hist)

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
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

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


ysg = savitzky_golay(y, window_size=31, order=4)

plt.plot(x, y, label='Original signal')
plt.plot(x, ysg, 'r', label='Filtered signal')
plt.legend()
plt.show()
