import os
import zipfile
import subprocess
from multiprocessing import Pool, cpu_count


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


def unwarp_decompress(arg, **kwarg):
    return decompress(*arg, **kwarg)


def decompress(input_path, output_path):
    try:
        command = ["gdcmconv", "-w", input_path, output_path]
        subprocess.call(command)
    except:
        pass


input_dcms, output_dcms = [], []
for d in os.listdir(unzipped_dir):
    sub_d = os.path.join(unzipped_dir, d)
    subsub_d = os.path.join(sub_d, os.listdir(sub_d)[0])
    for scan in os.listdir(subsub_d):
        input_scan_dir = os.path.join(subsub_d, scan)
        output_scan_dir = os.path.join(decompressed_dir, d, scan)
        create_dir(output_scan_dir)
        for dcm in os.listdir(input_scan_dir):
            input_dcms.append(os.path.join(input_scan_dir, dcm))
            output_dcms.append(os.path.join(output_scan_dir, dcm))

paras = zip(input_dcms, output_dcms)
pool = Pool(processes=cpu_count())
pool.map(unwarp_decompress, paras)
