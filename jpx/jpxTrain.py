########################################
##
from jpx import *
from jpxDataSource import *
from jpxModel import *

########################################
# train
setupLogger( );
logging.getLogger('jpx').info(f'start training')

ds = DataSource(dataPath=USE_DATA_PATH)
ds.printMemoryUssage('readDataSource')
#ds.readData(fileName=USE_TRAIN_FILENAME)
ds.readData(fileName=USE_TRAIN_FILENAME2)
ds.printMemoryUssage('readDataSource Done')

##################################
# hyper-parameter tuning
from sklearn.model_selection import ParameterGrid

TUNING = False

if(TUNING):
    DO_TRAIN_MODE = -1

    params = {
      'boosting_type': ['goss'],
      'objective': ['regression_l2'],
      'max_bin': [400],
      'num_leaves': [31, 50],
      'num_iterations': [1200],
      'learning_rate': [0.001, 0.003],
      'n_estimators': [2500],
      'min_child_samples': [100],
      'max_depth': [8]
    }

    param_grid = ParameterGrid(params)

    for i, param in enumerate(param_grid):
        lgm = ModelGeneric (dataSource=ds,name=MODEL_NAME_LGB, opt=param)
        lgm.train()
        print(i)
###################################

if DO_TRAIN_MODE >= 0:
    if (USE_MODEL_LGB):
        lgm = ModelGeneric (dataSource=ds,name=MODEL_NAME_LGB)
    if (USE_MODEL_XGB):
        xgb = ModelGeneric (dataSource=ds,name=MODEL_NAME_XGB)
    if (USE_MODEL_RGF):
        rgf = ModelGeneric (dataSource=ds,name=MODEL_NAME_RGF)
    if (USE_MODEL_NN):
        dnn = ModelNN (dataSource=ds,name=MODEL_NAME_DNN)
    if (USE_MODEL_NNM2):
        nnm2 = ModelGenericNN (dataSource=ds,name=MODEL_NAME_LGB)

    if (DO_TRAIN_MODE == 1) :
        if (USE_MODEL_LGB):
            lgm.trainRolling()
        if (USE_MODEL_XGB):
            xgb.trainRolling()
        if (USE_MODEL_NN):
            dnn.trainRolling()
    else:
        if (USE_MODEL_LGB) and not USE_MODEL_NNM2:
            # lgm.train()
            lgm.train2()
        if (USE_MODEL_XGB):
            xgb.train()
        if (USE_MODEL_RGF):
            rgf.train()
        if (USE_MODEL_NN):
            dnn.train()
        if (USE_MODEL_NNM2):
            nnm2.train()

logging.getLogger('jpx').info(f'End train execution')
