# Brain Tumor Classification
# Script for Settings
# Author: Qixun Qu
# Create on: 2017/09/10
# Modify on: 2017/10/06

#     ,,,         ,,,
#   ;"   ';     ;'   ",
#   ;  @.ss$$$$$$s.@  ;
#   `s$$$$$$$$$$$$$$$'
#   $$$$$$$$$$$$$$$$$$
#  $$$$P""Y$$$Y""W$$$$$
#  $$$$  p"$$$"q  $$$$$
#  $$$$  .$$$$$.  $$$$'
#   $$$DaU$$O$$DaU$$$'
#    '$$$$'.^.'$$$$'
#       '&$$$$$&'

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
FULL_SHAPE = [240, 240, 155, 4]
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


# Parameters for Keep Minimum Volume
EDGE_SPACE = 10


'''
Settings for Patches Generation
'''

# General Settings
CHANNELS = 4
TUMOT_MIN_SIZE = 500
PATCHES_FOLDER = "Patches"
RESIZE_FOLDER = "resize"
TUMOR_FOLDER = "tumor"
SHAPE_FILE = "shape.txt"
SHAPE_FILE_SPLIT = "\n"

# Values in Tumor Mask
NCRNET_MASK = 1  # Necrotic and the Non-Enhancing tumor
ED_MASK = 2      # the Peritumoral Edema
ET_MASK = 4      # Enhancing Tumor
ELSE_MASK = 0    # Everything Else

# Morphology type
MORPHOLOGY = ["original", "dilated", "eroded"]


'''
Settings for Data Augmentation
'''

# General Settings
AUGMENT_FOLDER = "Augmented"
LABEL_FILE = "labels.csv"
CASE_NO = "Case"
GRADE_LABEL = "Grade_Label"
GRADE_IV = 2
GRADE_III = 1
GRADE_II = 0
GRADE_UNKNOWN = -1
PARTIAL_SIZE = 49
