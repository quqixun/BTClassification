import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


radius_prop = 0.5
scan_types = ["flair", "t1", "t1ce", "t2", "seg"]

parent_dir = os.path.dirname(os.getcwd())
hgg_dir = "/home/user4/Desktop/Dataset/BraTS/HGG"
lgg_dir = "/home/user4/Desktop/Dataset/BraTS/LGG"

lgg_subjects = os.listdir(lgg_dir)

lgg_subj_0 = lgg_subjects[1]
lgg_subj_0_dir = os.path.join(lgg_dir, lgg_subj_0)

volume_paths = []
for t in scan_types:
    scan_name = lgg_subj_0 + "_" + t + ".nii.gz"
    volume_paths.append(os.path.join(lgg_subj_0_dir, scan_name))

volumes = []
if len(volume_paths) == len(scan_types):
    for path in volume_paths:
        volumes.append(np.rot90(nib.load(path).get_data(), 3))

new_mask = np.copy(volumes[-1])
new_mask[np.where(np.logical_and(new_mask != 4, new_mask != 1))] = 0
new_mask[np.where(new_mask != 0)] = 1

pos = np.where(new_mask == 1)
fp = np.array([np.min(pos[0]), np.min(pos[1]), np.min(pos[2])])
lp = np.array([np.max(pos[0]), np.max(pos[1]), np.max(pos[2])])
cp = list((fp + lp) // 2)

radius = np.min(np.abs(fp - lp)) // 2
distance = np.round(radius * radius_prop)
d = int(np.round(np.sqrt(distance ** 2 / 3)))

aug_pos = [[[cp[0] - d, cp[1] - d, cp[2] - d], [cp[0] + d, cp[1] + d, cp[2] + d]],
           [[cp[0] - d, cp[1] + d, cp[2] - d], [cp[0] + d, cp[1] - d, cp[2] + d]],
           [[cp[0] - d, cp[1] - d, cp[2] + d], [cp[0] + d, cp[1] + d, cp[2] - d]],
           [[cp[0] + d, cp[1] - d, cp[2] - d], [cp[0] - d, cp[1] + d, cp[2] + d]]]

pos_idx = np.random.randint(0, 4, 1)[0]
all_pos = [cp] + aug_pos[pos_idx]

# print(fp, lp, cp, distance)
# print(all_pos)

for pos in all_pos:
    name = "_".join(list(map(str, pos)))
    cor = np.rot90(volumes[2][pos[0], :, :], 1)
    sag = np.rot90(volumes[2][:, pos[1], :], 1)
    ax = volumes[2][:, :, pos[2]]

    plt.figure(num=name)
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.imshow(cor, cmap="gray")
    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.imshow(sag, cmap="gray")
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.imshow(ax, cmap="gray")

plt.show()
