import os
import zipfile
import subprocess

from tqdm import *
import pandas as pd


def create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return


parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, "data", "Asgeir UCSF", "Zipped DICOM")
unzipped_dir = os.path.join(parent_dir, "data", "Unzipped")
decompressed_dir = os.path.join(parent_dir, "data", "Dicompressed")

create_dir(unzipped_dir)
create_dir(decompressed_dir)

'''
Unzip file into folder
'''

# zips_list = os.listdir(data_dir)
# for z in zips_list:
#     z_name = z.split(".")[0]
#     z_path = os.path.join(data_dir, z)
#     z2path = os.path.join(unzipped_dir, z_name)
#     print("Preprocessing on " + z_path)
#     try:
#         with zipfile.ZipFile(z_path, "r") as zip_ref:
#             zip_ref.extractall(z2path)
#     except:
#         print("  Cannot unzip: " + z_path)


'''
Decompress DICOM files
'''

subjects = []
scans = []
undecomps = []

for d in tqdm(os.listdir(unzipped_dir)):
    sub_d = os.path.join(unzipped_dir, d)
    subsub_d = os.path.join(sub_d, os.listdir(sub_d)[0])
    for scan in os.listdir(subsub_d):
        input_scan_dir = os.path.join(subsub_d, scan)
        output_scan_dir = os.path.join(decompressed_dir, d, scan)
        create_dir(output_scan_dir)
        undecomp = 0
        for dcm in os.listdir(input_scan_dir):
            input_dcm = os.path.join(input_scan_dir, dcm)
            output_dcm = os.path.join(output_scan_dir, dcm)
            try:
                command = ["gdcmconv", "-w", input_dcm, output_dcm]
                subprocess.call(command)
            except:
                undecomp += 1
                continue
        subjects.append(d)
        scans.append(scan)
        undecomps.append(undecomp)

df = pd.DataFrame(data={"subject": subjects, "scans": scan, "log": undecomps})
df.to_csv("decompress_logs.csv", index=False)
