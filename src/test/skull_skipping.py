import os
import shutil
import subprocess
import pandas as pd
from multiprocessing import Pool, cpu_count


def unwrap_skull_stripping(arg, **kwarg):
    return skull_stripping(*arg, **kwarg)


def skull_stripping(case_name, subcase_name, subcase_path, target_dir):
    subcase_dir = os.path.join(target_dir, subcase_name)
    if os.path.isdir(subcase_dir):
        shutil.rmtree(subcase_dir)

    print "Skull-stripping on: ", case_name, subcase_name
    try:
        command1 = ["recon-all", "-subjid", subcase_name, "-i", subcase_path, "-autorecon1", "-sd", target_dir]
        cnull = open(os.devnull, 'w')
        subprocess.call(command1, stdout=cnull, stderr=subprocess.STDOUT)
    except:
        print "  Skull-skipping Failed: %s" % case_name, subcase_name
        shutil.rmtree(subcase_dir)
        return

    print "Convert to nii.gz of: ", case_name, subcase_name
    try:
        mgz_path = os.path.join(target_dir, subcase_name, "mri", "brainmask.mgz")
        nii_path = os.path.join(target_dir, subcase_name + ".nii.gz")
        command2 = ["mri_convert", mgz_path, nii_path]
        subprocess.call(command2, stdout=cnull, stderr=subprocess.STDOUT)
    except:
        print "  Failed to Convert %s" % case_name, subcase_name
        shutil.rmtree(subcase_dir)
        return

    shutil.rmtree(subcase_dir)
    return


def query(l, t, nt=[]):
    indices = []
    for i in range(len(l)):
        tnum = 0
        for it in t:
            if it in l[i].lower():
                tnum += 1

        ntnum = 0
        for n in nt:
            if n not in l[i].lower():
                ntnum += 1

        if tnum == len(t) and ntnum == len(nt):
            indices.append(int(i))

    scan_type = "_".join(t)
    print scan_type, ":", len(indices)

    return indices


total = pd.read_csv("total.csv")
des = total["info"].values.tolist()
subjectids = total["subject"].values.tolist()
scanids = total["scans"].values.tolist()

t1ax = query(des, ["t1", "ax"], ["flair", "spgr"])
t2ax = query(des, ["t2", "ax"], ["flair", "spgr"])
flairax = query(des, ["flair", "ax"], ["t1", "t2", "spgr"])


parent_dir = os.path.dirname(os.getcwd())
noskull_dir = os.path.join(parent_dir, "noskull_ax")
if not os.path.isdir(noskull_dir):
    os.makedirs(noskull_dir)

all_subjid = []
all_scanid = []
all_data_path = []
all_subj_dir = []
for indices, scan_type in zip([t1ax, t2ax, flairax], ["t1", "t2", "flair"]):
    scan_type_dir = os.path.join(noskull_dir, scan_type)
    if not os.path.isdir(scan_type_dir):
        os.makedirs(scan_type_dir)

    for index in indices:
        subjid = str(int(subjectids[index]))
        scanid = str(scanids[index])
        data_path = os.path.join(parent_dir, "new_data", subjid, "new", scanid + ".nii.gz")
        if not os.path.isfile(data_path):
            continue

        subj_dir = os.path.join(scan_type_dir, subjid)
        if not os.path.isdir(subj_dir):
            os.makedirs(subj_dir)

        all_subjid.append(subjid)
        all_scanid.append(scanid)
        all_data_path.append(data_path)
        all_subj_dir.append(subj_dir)

paras = zip(all_subjid, all_scanid, all_data_path, all_subj_dir)
pool = Pool(processes=cpu_count())
pool.map(unwrap_skull_stripping, paras)
