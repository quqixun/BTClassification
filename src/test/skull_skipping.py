import os
import shutil
import subprocess
from multiprocessing import Pool


# recon-all -subjid 0 -i 1.nii.gz -autorecon1 -sd /home/qixun/Desktop/new

def unwrap_skull_stripping(arg, **kwarg):
    return skull_stripping(*arg, **kwarg)


def skull_stripping(case_name, subcase_name, subcase_path, case_dir, target_dir):

    print "Skull-stripping on: ", case_name, subcase_name
    command1 = ["recon-all", "-subjid", subcase_name, "-i", subcase_path, "-autorecon1", "-sd", target_dir]
    cnull = open(os.devnull, 'w')
    subprocess.call(command1, stdout=cnull, stderr=subprocess.STDOUT)

    mgz_path = os.path.join(target_dir, "mri", "brainmask.mgz")
    nii_path = os.path.join(target_dir, subcase_name + ".nii.gz")
    command2 = ["mri_convert", mgz_path, nii_path]
    subprocess.call(command2, stdout=cnull, stderr=subprocess.STDOUT)
    return


parent_dir = os.path.dirname(os.getcwd())
noskull_dir = os.path.join(parent_dir, "noskull")
if not os.path.isdir(noskull_dir):
    os.makedirs(noskull_dir)

data_dir = os.path.join(parent_dir, "new_data")

for case in os.listdir(data_dir):
    case_dir = os.path.join(data_dir, case)
    case_noskull_dir = os.path.join(noskull_dir, case)
    if os.path.isdir(case_noskull_dir):
        shutil.rmtree(case_noskull_dir)
    os.makedirs(case_noskull_dir)

    subcases = os.listdir(case_dir)
    case_names = [case] * len(subcases)
    subcase_names = [subcase.split(".")[0] for subcase in subcases]
    subcase_paths = [os.path.join(case_dir, subcase) for subcase in subcases]
    case_dirs = [case_dir] * len(subcases)
    case_noskull_dirs = [case_noskull_dir] * len(subcases)

    paras = zip(case_names, subcase_names, subcase_paths, case_dirs, case_noskull_dirs)
    pool = Pool(processes=2)
    pool.map(unwrap_skull_stripping, paras)
