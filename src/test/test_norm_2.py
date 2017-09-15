import os
import numpy as np
from tqdm import *
from skimage import io
import matplotlib.pyplot as plt





PCTS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.998]




def _get_volume_landmarks(folder, dirs, target):
    
    volume_pct_ori = []
    for d in tqdm(dirs):
        path = os.path.join(folder, d, target + ".mha")
        volume = io.imread(path, plugin="simpleitk")
        volume = volume.astype(np.float32)

        volume_min = np.min(volume)
        if volume_min < 0:
            volume[np.where(volume == 0)] = volume_min
            volume = volume - np.min(volume)

        volume = volume[np.where(volume > 0)]
        sort = np.sort(volume)
        sort_len = len(sort)

        pct = []
        for p in PCTS:
            pct_idx = int(np.ceil(p * sort_len)) - 1
            if pct_idx < 0: pct_idx = 0
            pct.append(sort[pct_idx])

        volume_pct_ori.append(pct)

    volume_pct_ori = np.array(volume_pct_ori)
    landmarks = np.mean(volume_pct_ori, axis=0)

    return landmarks, volume_pct_ori




def _intensity_normalization(folder, dirs, target, landmarks, pct):

    for i in tqdm(range(len(dirs))):
        path = os.path.join(folder, dirs[i], target + ".mha")
        volume = io.imread(path, plugin="simpleitk")
        volume = volume.astype(np.float32)

        volume_min = np.min(volume)
        if volume_min < 0:
            volume[np.where(volume == 0)] = volume_min
            volume = volume - np.min(volume)

        bigger_idx = np.where(volume >= pct[i, -1])
        smaller_idx = np.where(volume < pct[i, 0])
        non_zero_idx = np.where(volume > 0)
        volume[non_zero_idx] = np.interp(volume[non_zero_idx], pct[i, :], landmarks)

        volume[bigger_idx] = landmarks[-1]
        volume[smaller_idx] = 0

        volume = volume.astype(np.int16)
        np.save("norm\\" + dirs[i] + ".npy", volume)
        volume = None

    return




def get_hist():
    hists = []
    for i in tqdm(range(220)):
        path = "E:\\ms\\data\\HGG\\" + str(i) + "\\" + str(i) + "_Flair.mha"
        volume = io.imread(path, plugin="simpleitk")
        hist = np.histogram(volume, bins=np.arange(1, 2001))
        hists.append(hist)
    return hists

def plot_hist(hists):
    plt.figure()
    for i in range(len(hists)):
        plt.plot(hists[i][1][1:], hists[i][0])
    plt.show()



if __name__ == "__main__":

    folder = "E:\\ms\\data\\HGG"
    target = "Flair"
    dirs = os.listdir(folder)

    landmarks, pct = _get_volume_landmarks(folder, dirs, target)
    _intensity_normalization(folder, dirs, target, landmarks, pct)
