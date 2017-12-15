import os
import subprocess
from multiprocessing import Pool, cpu_count


def unwarp_skull_stripping(arg, **kwarg):
    return skull_stripping(*arg, **kwarg)


def skull_stripping(nii_path, output_dir, scan_name, templates):
    print("Skull stripping on:", nii_path)
    try:
        output_prefix = os.path.join(output_dir, scan_name)
        command = ["antsBrainExtraction.sh", "-d 3", "-a", nii_path,
                   "-e", templates[0], "-m", templates[1], "-f", templates[2],
                   "-o", output_prefix]
        cnull = open(os.devnull, 'w')
        subprocess.call(command, stdout=cnull, stderr=subprocess.STDOUT)
        brain_name = output_prefix + "BrainExtractionBrain.nii.gz"
        os.rename(brain_name, output_prefix + ".nii.gz")
        os.remove(output_prefix + "BrainExtractionMask.nii.gz")
        os.remove(output_prefix + "BrainExtractionPrior0GenericAffine.mat")
    except:
        print("\tFailed on:", nii_path)
        return
    return


parent_dir = os.path.dirname(os.getcwd())
data_dir = os.path.join(parent_dir, "data", "Original", "LGG")
new_data_dir = os.path.join(parent_dir, "data", "NoSkull", "LGG")
if not os.path.isdir(new_data_dir):
    os.makedirs(new_data_dir)

nii_paths = []
output_dirs = []
scan_names = []

for subject in os.listdir(data_dir):
    subject_dir = os.path.join(data_dir, subject)
    if os.path.isdir(subject_dir):
        new_subject_dir = os.path.join(new_data_dir, subject)
        if not os.path.isdir(new_subject_dir):
            os.makedirs(new_subject_dir)
        for scan in os.listdir(subject_dir):
            scan_name = scan.split(".")[0]
            if "T1" in scan or "T2" in scan:
                nii_paths.append(os.path.join(subject_dir, scan))
                output_dirs.append(os.path.join(new_subject_dir))
                scan_names.append(scan_name)

templates_dir = os.path.join(parent_dir, "data", "Template")
templates = [os.path.join(templates_dir, "T_template0.nii.gz"),
             os.path.join(templates_dir, "T_template0_BrainCerebellumProbabilityMask.nii.gz"),
             os.path.join(templates_dir, "T_template0_BrainCerebellumRegistrationMask.nii.gz")]


paras = zip(nii_paths, output_dirs, scan_names, [templates] * len(nii_paths))
pool = Pool(processes=cpu_count())
pool.map(unwarp_skull_stripping, paras)
