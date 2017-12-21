import os
import warnings
import numpy as np
from tqdm import *
import nibabel as nib
import scipy.misc
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt


# Ignore the warning caused by SciPy
warnings.simplefilter("ignore", UserWarning)


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def trim_views(views):
    trimmed_views = []
    for v in views:
        non_bg_idx = np.where(v)
        rb, re = np.min(non_bg_idx[0]), np.max(non_bg_idx[0])
        cb, ce = np.min(non_bg_idx[1]), np.max(non_bg_idx[1])
        sub_view = v[rb:re + 1, cb:ce + 1]
        factors = [t / s for s, t in zip(sub_view.shape, TRIMMED_SIZE)]
        resized = zoom(sub_view, zoom=factors, order=1, prefilter=False)
        trimmed_views.append(resized)

    return trimmed_views


def extract_views(data_dir, views_dir, mode, norm=False, trim=False, trimmed_dir=None, volume_dir=None):
    print("Extract views to " + views_dir)
    subjects = os.listdir(data_dir)
    for subject in tqdm(subjects):
        subject_dir = os.path.join(data_dir, subject)
        volume_paths = []
        for t in SCAN_TYPES:
            scan_name = subject + "_" + t + ".nii.gz"
            volume_paths.append(os.path.join(subject_dir, scan_name))

        volumes = []
        if len(volume_paths) == len(SCAN_TYPES):
            for path in volume_paths:
                volume = np.rot90(nib.load(path).get_data(), 3)
                if norm:
                    if "seg" not in path:
                        volume = volume / np.max(volume)
                volumes.append(volume)
        else:
            print(subject + " does not have 4 types volumes.")
            continue

        new_mask = np.copy(volumes[-1])
        new_mask[np.where(np.logical_and(new_mask != 4, new_mask != 1))] = 0
        new_mask[np.where(new_mask != 0)] = 1

        pos = np.where(new_mask == 1)
        fp = np.array([np.min(pos[0]), np.min(pos[1]), np.min(pos[2])])
        lp = np.array([np.max(pos[0]), np.max(pos[1]), np.max(pos[2])])
        cp = list((fp + lp) // 2)

        all_pos = [cp]
        if mode == "lgg":
            radius = np.min(np.abs(fp - lp)) // 2
            distance = np.round(radius * RADIUS_PROP)
            d = int(np.round(np.sqrt(distance ** 2 / 3)))

            aug_pos = [[[cp[0] - d, cp[1] - d, cp[2] - d], [cp[0] + d, cp[1] + d, cp[2] + d]],
                       [[cp[0] - d, cp[1] + d, cp[2] - d], [cp[0] + d, cp[1] - d, cp[2] + d]],
                       [[cp[0] - d, cp[1] - d, cp[2] + d], [cp[0] + d, cp[1] + d, cp[2] - d]],
                       [[cp[0] + d, cp[1] - d, cp[2] - d], [cp[0] - d, cp[1] + d, cp[2] + d]]]

            pos_idx = np.random.randint(0, 4, 1)[0]
            all_pos += aug_pos[pos_idx]

        counter = 0
        for pos in all_pos:
            cor_volume = np.zeros([155, 240, 4])
            sag_volume = np.zeros([155, 240, 4])
            ax_volume = np.zeros([240, 240, 4])
            for i in range(len(SCAN_TYPES[:-1])):
                cor = np.rot90(volumes[i][pos[0], :, :], 1)
                sag = np.rot90(volumes[i][:, pos[1], :], 1)
                ax = volumes[i][:, :, pos[2]]

                cor_volume[..., i] = cor
                sag_volume[..., i] = sag
                ax_volume[..., i] = ax

                save_dir = os.path.join(views_dir, subject, SCAN_TYPES[i])
                create_dir(save_dir)

                views = [cor, sag, ax]
                for v, t in zip(views, VIEW_TYPES):
                    save_path = os.path.join(save_dir, t + "_" + str(counter) + ".npy")
                    png_path = os.path.join(save_dir, t + "_" + str(counter) + ".png")
                    np.save(save_path, v)
                    scipy.misc.imsave(png_path, v)

                if trim:
                    save_dir = os.path.join(trimmed_dir, subject, SCAN_TYPES[i])
                    create_dir(save_dir)
                    trimmed_views = trim_views(views)
                    for v, t in zip(trimmed_views, VIEW_TYPES):
                        save_path = os.path.join(save_dir, t + "_" + str(counter) + ".npy")
                        png_path = os.path.join(save_dir, t + "_" + str(counter) + ".png")
                        np.save(save_path, v)
                        scipy.misc.imsave(png_path, v)

            views_volume = [cor_volume, sag_volume, ax_volume]
            subject2dir = os.path.join(volume_dir, subject, str(counter))
            create_dir(subject2dir)
            for vv, t in zip(views_volume, VIEW_TYPES):
                save_path = os.path.join(subject2dir, t + ".npy")
                np.save(save_path, vv.astype(np.int16))

            counter += 1

            # plt.figure()
            # plt.subplot(2, 2, 1)
            # plt.imshow(cor_volume[..., 0], cmap="gray")
            # plt.subplot(2, 2, 2)
            # plt.imshow(cor_volume[..., 1], cmap="gray")
            # plt.subplot(2, 2, 3)
            # plt.imshow(cor_volume[..., 2], cmap="gray")
            # plt.subplot(2, 2, 4)
            # plt.imshow(cor_volume[..., 3], cmap="gray")
            # plt.show()

    return


RADIUS_PROP = 0.5
TRIMMED_SIZE = [110, 110]
SCAN_TYPES = ["flair", "t1", "t1ce", "t2", "seg"]
VIEW_TYPES = ["cor", "sag", "ax"]

parent_dir = os.path.dirname(os.getcwd())
hgg_dir = "/home/user4/Desktop/Dataset/BraTS/HGG"
lgg_dir = "/home/user4/Desktop/Dataset/BraTS/LGG"

new_hgg_dir = "/home/user4/Desktop/btc/data/Original/BraTS/HGGViews"
new_lgg_dir = "/home/user4/Desktop/btc/data/Original/BraTS/LGGViews"
create_dir(new_hgg_dir)
create_dir(new_lgg_dir)

trim_hgg_dir = "/home/user4/Desktop/btc/data/Original/BraTS/HGGTrimmedViews"
trim_lgg_dir = "/home/user4/Desktop/btc/data/Original/BraTS/LGGTrimmedViews"
create_dir(trim_hgg_dir)
create_dir(trim_lgg_dir)

hgg_vol_dir = "/home/user4/Desktop/btc/data/Original/BraTS/HGGViewsVolume"
lgg_vol_dir = "/home/user4/Desktop/btc/data/Original/BraTS/LGGViewsVolume"
create_dir(hgg_vol_dir)
create_dir(lgg_vol_dir)

extract_views(data_dir=hgg_dir, views_dir=new_hgg_dir, mode="hgg", norm=False,
              trim=True, trimmed_dir=trim_hgg_dir, volume_dir=hgg_vol_dir)
extract_views(data_dir=lgg_dir, views_dir=new_lgg_dir, mode="lgg", norm=False,
              trim=True, trimmed_dir=trim_lgg_dir, volume_dir=lgg_vol_dir)
