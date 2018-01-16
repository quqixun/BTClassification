import os
import numpy as np
import nibabel as nib


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def load_nii(path):
    return nib.load(path).get_data()


def save_nii(data, path):
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    return


pos_table = [["30009", [100, 160], [86, 154], [21, 81]],
             ["30010", [73, 103], [128, 162], [48, 81]],
             ["30018", [24, 68], [54, 101], [62, 134]],
             ["30023", [43, 71], [121, 142], [50, 74]],
             ["30024", [51, 74], [155, 177], [52, 73]],
             ["30040", [47, 75], [25, 51], [52, 80]],
             ["30041", [29, 76], [100, 136], [101, 148]],
             ["30043", [61, 77], [129, 155], [111, 144]],
             ["30047", [111, 142], [106, 131], [117, 144]],
             ["30060", [45, 68], [111, 122], [117, 145]],
             ["30088", [74, 97], [134, 186], [93, 127]],
             ["30089", [74, 97], [134, 186], [93, 127]],
             ["30094", [78, 97], [126, 170], [61, 91]],
             ["70003", [40, 77], [135, 163], [47, 89]],
             ["70010", [43, 83], [131, 182], [102, 144]],
             ["70011", [75, 97], [137, 170], [104, 134]],
             ["70016", [60, 83], [106, 136], [46, 77]],
             ["70034", [33, 66], [107, 143], [87, 128]],
             ["70045", [26, 47], [115, 147], [85, 119]],
             ["70047", [108, 128], [133, 160], [109, 137]],
             ["70048", [33, 74], [93, 130], [77, 118]],
             ["70055", [37, 98], [112, 193], [72, 149]],
             ["70060", [106, 137], [160, 197], [76, 115]],
             ["70103", [25, 91], [43, 112], [59, 146]],
             ["70108", [65, 100], [78, 106], [59, 95]],
             ["30030", [43, 76], [93, 125], [31, 68]]]


# ["30053", [], [], []],
# ["70024", [], [], []],

in_dir = "/home/user4/Desktop/BrainPrep/FlairT1ceN4BFC/"
out_dir = "/home/user4/Desktop/BrainPrep/TumorSegment/"

for pos in pos_table:
    print(pos[0])
    subject = pos[0]
    x1, x2 = pos[1][0], pos[1][1]
    y1, y2 = pos[2][0], pos[2][1]
    z1, z2 = pos[3][0], pos[3][1]

    flair_path = in_dir + subject + "/flair.nii.gz"
    t1ce_path = in_dir + subject + "/t1ce.nii.gz"

    out_flair_path = out_dir + subject + "/flair.nii.gz"
    out_t1ce_path = out_dir + subject + "/t1ce.nii.gz"

    flair = load_nii(flair_path)
    t1ce = load_nii(t1ce_path)
    mask = np.ones(flair.shape) * 0.333
    mask[x1:x2, y1:y2, z1:z2] = 1.

    flair = np.multiply(flair, mask)
    save_nii(flair, out_flair_path)

    t1ce = np.multiply(t1ce, mask)
    save_nii(t1ce, out_t1ce_path)

    mask_path = out_dir + subject + "/mask.nii.gz"
    mask = (mask == 1).astype(np.uint8)
    save_nii(mask, mask_path)
