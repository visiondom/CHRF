from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 2

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = ""

# Path (a file path, or URL like detectron2://.., https://..) to a checkpoint file
# to be loaded to the model. You can find available models in the model zoo.
_C.MODEL.WEIGHTS = ""

_C.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
_C.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

# -----------------------------------------------------------------------------
# INPUT
# Augmentation args
# -----------------------------------------------------------------------------
_C.INPUT = CN()

# image format: "BGR", "LAB", "RGB"
_C.INPUT.FORMAT = "RGB"

# rotate angel range for reference image
_C.INPUT.ROTATE_ANGLE = [-0.0, 0.0]

# full size for target image
_C.INPUT.FULL_SIZE = 512
_C.INPUT.FULL_SIZE_TEST = 512
_C.INPUT.MIN_SIZE_TRAIN = 640
_C.INPUT.MAX_SIZE_TRAIN = 3000
_C.INPUT.DIVISIBILITY = 1

# crop type for cropping patch(while preparing data)
_C.INPUT.CROP_TYPE = "absolute"
# reference patch size
_C.INPUT.PATCH_SIZE = [448,448]
_C.INPUT.PATCH_SIZE_TEST = [448,448]

# resize
_C.INPUT.MIN_SIZE_TEST = 480
_C.INPUT.MAX_SIZE_TEST = 3000


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training. Must be registered in DatasetCatalog
_C.DATASETS.TRAIN = ()
_C.DATASETS.TRAINSET_NUM = 5994
# List of the dataset names for test. Must be registered in DatasetCatalog
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"
# Freeze the first several stages so they are not trained.
# There are 5 stages in ResNet. The first is a convolution, and the following
# stages are each group of residual blocks.
_C.MODEL.BACKBONE.FREEZE_AT = 0
_C.MODEL.BACKBONE.POST_CONV = False
_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.NAME = ""
_C.MODEL.HEAD.SCALE = 20.0
_C.MODEL.HEAD.IN_DROP = 0.0
_C.MODEL.HEAD.CLS_BIAS = True
# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

# set default depth to 50
_C.MODEL.RESNETS.DEPTH = 50
_C.MODEL.RESNETS.OUT_FEATURES = [
    "res5"
]  # res4 for C4 backbone, res2..5 for FPN backbone

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.RESNETS.NORM = "BN"

# Baseline width of each group.
# Scaling this parameters will scale the width of all bottleneck layers.
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1

# Output width of res2. Scaling this parameters will scale the width of all 1x1 convs in ResNet
# For R18 and R34, this needs to be set to 64
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

# Apply Deformable Convolution in stages
# Specify if apply deform_conv on Res2, Res3, Res4, Res5
_C.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]
# Use True to use modulated deform_conv (DeformableV2, https://arxiv.org/abs/1811.11168);
# Use False for DeformableV1.
_C.MODEL.RESNETS.DEFORM_MODULATED = False
# Number of groups in deformable conv.
_C.MODEL.RESNETS.DEFORM_NUM_GROUPS = 1


# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.MULTIHRESNETS = CN(new_allowed=True)
_C.MODEL.MULTIHRESNETS.STAGE_DIVIDE_AT = 4
_C.MODEL.MULTIHRESNETS.SHARED_FREEZE_AT = 3
_C.MODEL.MULTIHRESNETS.HIER_FREEZE_AT = [0, 0, 4, 4] # order for ['class', 'genera', 'family', 'order']

# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN()
# Names of the input feature maps to be used by FPN
# They must have contiguous power of 2 strides
# e.g., ["res2", "res3", "res4", "res5"]
_C.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
_C.MODEL.FPN.OUT_CHANNELS = 256

# Options: "" (no norm), "GN"
_C.MODEL.FPN.NORM = ""

# Types for fusing the FPN top-down and lateral features. Can be either "sum" or "avg"
_C.MODEL.FPN.FUSE_TYPE = "sum"

# ---------------------------------------------------------------------------- #
# Additional OPTIONS
# new allowed for debugging and fixing parameters
# ---------------------------------------------------------------------------- #

_C.MODEL.NECK = CN(new_allowed=True)
_C.MODEL.NECK.NAME = ""
_C.MODEL.NECK.OUT_CHANNELS = 2048
_C.MODEL.NECK.FUSION_MODE = "concat"  # "concat", "add"


_C.BENCHMARK = CN(new_allowed=True)
_C.BENCHMARK.HIERARCHY = ['class', 'genus', 'family', 'order']  # default CUB
_C.BENCHMARK.IN_FEATURES = ["res5"]
_C.BENCHMARK.CLASSIFICATION = CN()
_C.BENCHMARK.CLASSIFICATION.CATEGORY_NUM = 1000
_C.BENCHMARK.CLASSIFICATION.POOLING_MODE = "avg"
# currently supported:
# [ "product_sum","fmap-gmap_cat","mask_cat" ]
_C.BENCHMARK.GATTENTION_MODE = "product_sum"
# currently supported:
# [ "coarse","mid","fg","hybrid" ]
_C.BENCHMARK.GATTENTION_LEVEL = "coarse"
_C.BENCHMARK.LOAD_GAZE = False
_C.BENCHMARK.KEEP_SAME_GAZE_NUM = True
_C.BENCHMARK.F_WEIGHTS = 1.0
_C.BENCHMARK.G_WEIGHTS = 0.0

_C.BENCHMARK.OrthRegRegularization = "COR"  # "OR", "COR", "C"
_C.BENCHMARK.ORR_LAMBDA = 1.0
_C.BENCHMARK.SIGMA = [0.4, 0.1, 0.01]
_C.BENCHMARK.CENTER_BETA = 5e-2

# ---------------------------------------------------------------------------- #
# Multi-Hierarchy Benchmark
# ---------------------------------------------------------------------------- #
_C.MHBENCHMARK = CN(new_allowed=True)
_C.MHBENCHMARK.CLASSIFICATION = CN()
_C.MHBENCHMARK.CLASSIFICATION.CATEGORY_NUM = [200, 122, 37, 13] # order for ['class', 'genera', 'family', 'order']

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# See detectron2/solver/build.py for LR scheduler options
_C.SOLVER.LR_SCHEDULER_NAME = "ContinuousExponentialLR"

# optimizer, "SGD", "ADAM", "ADAMW", "ADAGRAD"
_C.SOLVER.OPTIM = "ADAM"

_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.NESTEROV = False

_C.SOLVER.WEIGHT_DECAY = 0.00001
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

# lr schedule
_C.SOLVER.MAX_ITER = 40000
_C.SOLVER.GAMMA = 0.9
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 200
_C.SOLVER.WARMUP_ITERS = 200
_C.SOLVER.WARMUP_METHOD = "linear"

# Save a checkpoint after every this number of iterations
_C.SOLVER.CHECKPOINT_PERIOD = 5000

# Number of images per batch across all machines.
# If we have 16 GPUs and IMS_PER_BATCH = 32,
# each GPU will see 2 images per batch.
# May be adjusted automatically if REFERENCE_WORLD_SIZE is set.
_C.SOLVER.IMS_PER_BATCH = 8

# The reference number of workers (GPUs) this config is meant to train with.
# With a non-zero value, it will be used by DefaultTrainer to compute a desired
# per-worker batch size, and then scale the other related configs (total batch size,
# learning rate, etc) to match the per-worker batch size if the actual number
# of workers during training is different from this reference.
_C.SOLVER.REFERENCE_WORLD_SIZE = 0

# Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
# biases. This is not useful (at least for recent models). You should avoid
# changing these and they exist only to reproduce Detectron v1 training if
# desired.

_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY

# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

# For compatible with Epoch based setting
_C.SOLVER.EPOCH_BASE = 1

# For ContinuousExponentialLR
_C.SOLVER.BASE_DURATION = 2.0  # set _C.SOLVER.GAMMA = 0.9

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

# The period (in terms of steps) for evaluation at train time.
# Set to 0 to disable. It may slow down the training speed if the evaluation
# process is time comsuming.
_C.TEST.EVAL_PERIOD = 0

# for evaluation
_C.TEST.IMS_PER_PROC = 1

_C.TEST.MODE = "classify"

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Directory where output files are written
_C.OUTPUT_DIR = "./output"
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed increases
# reproducibility but does not guarantee fully deterministic behavior.
# Disabling all parallelism further increases reproducibility.
_C.SEED = -1
# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = False
# The period (in terms of steps) for minibatch visualization at train time.
# Set to 0 to disable.
_C.VIS_PERIOD = 0
