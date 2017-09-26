# Brain Tumor Classification
# Script for Settings
# Author: Qixun Qu
# Create on: 2017/09/10
# Modify on: 2017/09/25

'''

Basic Settings for Classification Process

--- Reorganization
--- Preprocessing
--- Patches Generation
--- Build Models
--- Train Models
--- ...

'''


'''
Settings for Reorganization
'''

MASK_NAME = "mask"
ORIGINAL_FOLDER = "Original"
SOURCE_EXTENSION = ".nii.gz"
NON_USEFUL_VOLUME = "GlistrBoost"
REPLACE_MASK_NAME = "ManuallyCorrected"
VOLUME_TYPES = ["flair", "t1", "t1Gd", "t2"]


'''
Settings for Preprocessing
'''

# General Settings
DATA_FOLDER = "data"
TEMP_FOLDER = "Temp"
MASK_FOLDER = "mask"
FULL_FOLDER = "full"
TARGET_EXTENSION = ".npy"
PREPROCESSED_FOLDER = "Preprocessed"
BRAIN_SHAPE = [240, 240, 155]


# Parameters for N4BiasFieldCorrection
N4_BSPLINE = 300
N4_DIMENSION = 3
N4_THRESHOLD = 1e-4
N4_SHRINK_FACTOR = 5
N4_ITERATION = [100, 100, 60, 40]


# Parameters for Intensity Normalization
PCTS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.998]
PCTS_COLUMNS = [str(p) for p in PCTS]


'''
Settings for Patches Generation
'''

# General Settings
CHANNELS = 4
TUMOR_FOLDER = "Tumor"
PATCH_SIZE = [33, 33, 33, 4]
