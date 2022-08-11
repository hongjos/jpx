USE_MODEL_NN  = False
USE_MODEL_LGB = True
USE_MODEL_RGF = False
USE_MODEL_XGB = False
USE_MODEL_SVR = False # too slow

USE_MODEL_NNM2 = False # use nn encoder as feature
if USE_MODEL_NNM2:
    USE_MODEL_NN  = False
    USE_MODEL_LGB = True
    USE_MODEL_RGF = False
    USE_MODEL_XGB = False
    USE_MODEL_SVR = False

DO_TRAIN_MODE = 0 # 1 - rolling training; else training


USE_MODEL_NN_TRAIN_EPOCH = 10