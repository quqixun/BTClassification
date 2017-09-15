import numpy as np
from tqdm import *
from skimage import io
import matplotlib.pyplot as plt





IMIN = 1
IMAX = 4000
PCTS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.998]



def _interp(pct, imin, imax):
    
    pct = np.array(pct).astype(np.float32)

    if len(np.unique(pct)) == 1:
    	return [imax] * len(pct)

    pct_diff = pct - np.min(pct)
    pct_gap = np.max(pct) - np.min(pct)
    temp = (imax-imin) * (pct_diff) / (pct_gap) + imin
    
    return list(temp)



def _get_volume_landmark(volume):

    image_idx = []
    image_pct = []
    volume_pct = []

    for i in range(volume.shape[0]):
        image = volume[i, :, :]
        image = image[np.where(image > 0)]

        if len(image) > 0:
            image_idx.append(i)
            sort = np.sort(image)
            pct = []

            for p in range(len(PCTS)):
                pct_idx = int(np.ceil(PCTS[p] * len(sort))) - 1
                if pct_idx < 0:
                    pct_idx = 0
                pct.append(sort[pct_idx])

            image_pct.append(pct)
            volume_pct.append(_interp(pct, IMIN, IMAX))
	        
    volume_pct = np.array(volume_pct)
    # print(volume_pct.shape)
    landmarks = np.mean(volume_pct, axis=0)

    return image_idx, np.array(image_pct), landmarks




def _intensity_normalization(volume):
    volume = volume.astype(np.float32)
    image_idx, image_pct, landmarks = _get_volume_landmark(volume)

    for i in range(len(image_idx)):        

        image = volume[image_idx[i], :, :]
        idx = np.where(image > image_pct[i, -1])
        image[idx] = IMAX

        for p in range(len(PCTS) - 1):
            if image_pct[i, p] != image_pct[i, p+1]:
                idx = np.where(np.logical_and(image > image_pct[i, p],
        	                                  image <= image_pct[i, p+1]))
                image[idx] = _interp(image[idx], landmarks[p], landmarks[p+1])
            else:
            	idx = np.where(image == image_pct[i, p+1])
            	image[idx] = landmarks[p+1]

        volume[image_idx[i], :, :] = image

    return volume



def get_hist():

    hists = []
    for i in tqdm(range(220)):
        # i = 45
        # path = "E:\\ms\\data\\HGG\\" + str(i) + "\\T2.mha"
        # volume = io.imread(path, plugin="simpleitk")
        path = "norm\\" + str(i) + ".npy"
        volume = np.load(path)
        # volume = volume.astype(np.int16)

        volume_min = np.min(volume)
        if volume_min < 0:
            print("here")
            volume[np.where(volume == 0)] = volume_min
            volume = volume - np.min(volume)

        hist = np.histogram(volume, bins=np.arange(1, 1000))
        hists.append(hist)
    return np.array(hists)

def plot_hist(hists):
    idx = np.arange(219, 1, -1)
    plt.figure()
    plt.title("FLAIR")
    for i in idx:
        x = hists[i, 1][1:]
        y = hists[i, 0]
        i = np.where(y > 0)
        x = x[i]
        y = y[i]
        plt.plot(x, y)
    plt.show()



if __name__ == "__main__":

    hists = get_hist()
    # print(hists.shape)
    plot_hist(hists)

# for i in tqdm(range(220)):
# path = "E:\\ms\\data\\HGG\\" + str(i) + "\\" + str(i) + "_Flair.mha"
# volume = io.imread(path, plugin="simpleitk")
# volume = _intensity_normalization(volume)
# np.save("norm\\" + str(i) + ".npy", volume)
