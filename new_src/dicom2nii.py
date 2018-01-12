from __future__ import print_function

import os
import dicom
import subprocess
from multiprocessing import Pool, cpu_count


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def unwarp_dcm2nii(arg, **kwarg):
    return dcm2nii(*arg, **kwarg)


def dcm2nii(in_path, out_path):
    try:
        print("Working on: ", in_path)
        devnull = open(os.devnull, "w")
        command = ["dcm2nii", "-o", out_path, "-d", "N", "-e", "N", "-i", "Y", in_path]
        subprocess.call(command, stdout=devnull, stderr=subprocess.STDOUT)

        for out_file in os.listdir(out_path):
            if out_file[0] == "o" or out_file[0:2] == "co":
                os.remove(os.path.join(out_path, out_file))
    except IOError:
        print("\tFailed on: ", in_path)
    return


# data_dir = os.path.join(os.getcwd(), "Data", "Paris")
# new_data_dir = os.path.join(os.getcwd(), "Data2Nii", "Paris2nii")
# create_dir(new_data_dir)

data_dir = os.path.join(os.getcwd(), "Data", "UCSF")
new_data_dir = os.path.join(os.getcwd(), "Data2Nii", "UCSF2nii")
create_dir(new_data_dir)

subject_names = os.listdir(data_dir)
scan_dirs, new_scan_dirs = [], []
for subject_name in subject_names:
    subject_dir = os.path.join(data_dir, subject_name)
    if not os.path.isdir(subject_dir):
        continue

    new_subject_dir = os.path.join(new_data_dir, subject_name)
    create_dir(new_subject_dir)

    scan_names = os.listdir(subject_dir)
    for scan_name in scan_names:

        if "._" in scan_name:
            continue

        scan_dir = os.path.join(subject_dir, scan_name)

        if len(os.listdir(scan_dir)) == 0:
            print(scan_dir)
            continue

        one_dcm_file = os.path.join(scan_dir, os.listdir(scan_dir)[0])
        dcm = dicom.read_file(one_dcm_file, force=True)

        try:
            new_scan_name = scan_name + " " + dcm.SeriesDescription
        except ValueError:
            new_scan_name = scan_name

        new_scan_dir = os.path.join(new_subject_dir, new_scan_name)
        create_dir(new_scan_dir)

        scan_dirs.append(scan_dir)
        new_scan_dirs.append(new_scan_dir)

paras = zip(scan_dirs, new_scan_dirs)
pool = Pool(processes=cpu_count())
pool.map(unwarp_dcm2nii, paras)
