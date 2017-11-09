# Brain Tumor Classification
# Script for Settings
# Author: Qixun Qu
# Create on: 2017/09/10
# Modify on: 2017/11/09

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
PCTS = [0, 0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 0.998]
PCTS_COLUMNS = [str(p) for p in PCTS]


# Parameters for Keep Minimum Volume
EDGE_SPACE = 10


'''
Settings for Patches or Volumes Generation
'''

# General Settings
CHANNELS = 4
TUMOT_MIN_SIZE = 500
PATCHES_FOLDER = "Patches"
VOLUMES_FOLDER = "Volumes"
SLICES_FOLDER = "Slices"
VOLUME_SHAPE = [112, 112, 88, CHANNELS]
VOLUME_ONE_CHANNEL_SHAPE = [112, 112, 88, 1]
SLICE_SHAPE = [112, 112, CHANNELS]
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
MORP_ITER_NUM = 5
ERODABLE_THRESH = 35

# Resize the tumor
ZOOM_ORDER = 3
ZOOM_FILTER = False

# Threshold for Extracting Slices
MASK_THRESHOLD = ET_MASK
PROP_THRESHOLD = 0.2
PROP_NON_BG = 0.25


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

SCOPE_MIN = 5
SCOPE_MAX = 10
# PARTIAL_NUM = 5
GRADE_II_PARTIALS = 7
GRADE_III_PARTIALS = 6
GRADE_IV_PARTIALS = 4


'''
Settings for Class of BTCTFRecords
'''

# Create TFRecords
RANDOM_SEED = 0  # 0, 1, 2, ...
PROPORTION = 0.5
TFRECORDS_FOLDER = "TFRecords"
GRADES_LIST = [GRADE_II, GRADE_III, GRADE_IV]
TRAIN_SET_FILE = "train_set.txt"
VALIDATE_SET_FILE = "validate_set.txt"
DATA_NUM_FILE = "data_num.json"
CASES_FILE_SPLIT = "\n"
TFRECORD_TRAIN = "train.tfrecord"
TFRECORD_VALIDATE = "validate.tfrecord"
TRAIN_MODE = "train"
VALIDATE_MODE = "validate"

# Decode TFRecords
PATCH_SHAPE = [PARTIAL_SIZE] * 3 + [CHANNELS]
NUM_THREADS = 4


'''
Settings for Printing
'''

PCW = "\033[0;0m"
PCR = "\033[1;31m"
PCG = "\033[32m"
PCY = "\033[1;33m"
PCB = "\033[1;34m"
PCP = "\033[1;35m"
PCC = "\033[1;36m"


'''
Settings of Models' Names
'''

CNN = "cnn"
FULL_CNN = "full_cnn"
RES_CNN = "res_cnn"
DENSE_CNN = "dense_cnn"
CAE_STRIDE = "cae_stride"
CAE_POOL = "cae_pool"
