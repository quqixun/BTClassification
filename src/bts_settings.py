# Brain Tumor Segmentation
# Script for Settings
# Author: Qixun Qu
# Date: 2017/09/10

'''

Basic Settings for Segmentation Process

--- Reorganization
--- Preprocessing
--- Patches Generation
--- Build Models
--- Parameters for Training Models
--- ...

'''



'''
Settings for Reorganization
'''
VOLUME_NAMES = ["Flair", "T1", "T1c", "T2", "Mask"]


'''
Settings for Preprocessing
'''

# Volume Shape
VOLUME_SHAPE = [155, 240, 240]
BRAIN_SHAPE = [155, 240, 240, 4]


# Parameters for N4BiasFieldCorrection
N4_BSPLINE = 300
N4_THRESHOLD = 1e-4
N4_SHRINK_FACTOR = 5
N4_ITERATION = [100, 100, 60, 40]


'''
Settings for Patches Generation
'''
PATCH_SIZE = [64, 64, 64, 4]  # [32, 32, 32, 4]
