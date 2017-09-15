import os
from tqdm import tqdm
from time import sleep
from shutil import copy



target = "E:\\ms\\data"
source = "E:\\Data\\Brain\\BRATS2015_Training"

sub_target = ["HGG", "LGG"]
names = ["Flair", "T1", "T1c", "T2", "Mask"]


for st in sub_target:
	print("Starting copy files in ", st)
	to = os.path.join(target, st)
	path = os.path.join(source, st)
	dirs = os.listdir(path)
	n = 0
	for d in tqdm(dirs):
		sub_path = os.path.join(path, d)
		sub_dirs = os.listdir(sub_path)
		sub_to = os.path.join(to, str(n))
		if not os.path.exists(sub_to):
			os.makedirs(sub_to)
		i = 0
		for sd in sub_dirs:
			sub_sub_path = os.path.join(sub_path, sd)
			source_files = os.listdir(sub_sub_path)

			for sf in source_files:
				if sf.endswith(".mha"):
					source_file = sf

			source_path = os.path.join(sub_sub_path, source_file)

			target_file = str(n) + "_" + names[i] + ".mha"
			target_path = os.path.join(sub_to, target_file)
			copy(source_path, target_path)
			i += 1

			sleep(0.1)
		n += 1

print("Done")
