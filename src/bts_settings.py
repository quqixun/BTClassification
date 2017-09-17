# Brain Tumor Segmentation
# Script for Settings
# Author: Qixun Qu
# Create on: 2017/09/10
# Modify on: 2017/09/17

'''

Basic Settings for Segmentation Process

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
VOLUME_NAMES = ["Flair", "T1", "T1c", "T2"]


'''
Settings for Preprocessing
'''

# General Settings
FULL_SHAPE = [155, 240, 240, 4]


# Parameters for N4BiasFieldCorrection
N4_BSPLINE = 300
N4_DIMENSION = 3
N4_THRESHOLD = 1e-4
N4_SHRINK_FACTOR = 5
N4_ITERATION = [100, 100, 60, 40]


# Parameters for Intensity Normalization
PCTS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.998]


'''
Settings for Patches Generation
'''
PATCH_SIZE = [33, 33, 33, 4]
