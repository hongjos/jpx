########################################
from jpxDataSource import *
from jpxModel import *

from jpxDataSource import *
from jpxModel import *

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

    
####################################
class ModelGenericTest(ModelGeneric):
    def __init__(self, dataSource=None, name='model', opt=None):
        super(ModelGenericTest, self).__init__(dataSource=dataSource, name=name, opt=opt)

    ################
    def getCVIterator(self, df, nFolds=4):
        cviterator = []
        num = df.shape[0] 
        head = int(num*50/100) ;
        tail = num - head ;
        tnum = int(tail / (nFolds+1)) ;
        snum = int(tail / (nFolds+1)) ;
        for i in range(nFolds):
            n = int(head + tail * i / nFolds) ;
            n0 = max(n-tnum, 0)
            m = min(n + snum, df.shape[0]) ;
            trainIndices = np.arange(n0, n+1)
            testIndices =  np.arange(n+1, m)
            cviterator.append( (trainIndices, testIndices) )
        return cviterator

    ################
    # cross validation train
    def trainCV(self, dataSource=None, opt=None):
        if (self.name == MODEL_NAME_LGB):
            #parameters = {
            #    'boosting_type' : ['gbdt', 'dart'], 
            #    'num_leaves': [5, 11, 17, 23, 35],
            #    'learning_rate': [0.005],
            #    'n_estimators': [200],
            #    'reg_alpha': [0]
            #}
            parameters = {
              'boosting_type': ['goss'],
              'objective': ['regression_l2'],
              'max_bin': [250],
              'num_leaves': [250],
              'num_iterations': [100, 200, 500, 1000],
              'learning_rate': [0.01, 0.001, 0.0001],
              'n_estimators': [2500],
              'min_child_samples': [100],
              'max_depth': [9]
            }       
        elif (self.name == MODEL_NAME_XGB):
            parameters = {
                'n_estimators': [2000],
                'booster': ['dart'],
                'max_depth': [14, 16],
                'eta': [0.04],
                'min_child_weight': [25, 50],
                'gamma': [100.0],
                'lambda': [100],
                'tree_method': ['gpu_hist'],
                'verbosity': [2]
            }
        else:
            parameters = {
            }
        self.ds.printMemoryUssage('train')
        tmline = np.sort(self.ds.raw['Date'].unique());
        idx = int(tmline.shape[0] * 30 / 100) ;
        X, Y = self.ds.getFeature(start=0,end=tmline[idx]) 

        self.logger.info(f'######### model {self.name} cv' ) ;
        for key in self.model:
            self.logger.info(f"GridSearchCV for: {key}")
            grid_search = GridSearchCV(
                estimator=self.buildModel(X, Y), # for cv no opt provided
                param_grid=parameters,
                n_jobs = -1,
                cv = self.getCVIterator(X, 1), #cv = 5,
                verbose=True
            )
            grid_search.fit(X, Y)
            self.model[self.name] = grid_search.best_estimator_
            self.modelOpt[self.name] = grid_search.best_params_ ;
            self.logger.info(f'######### model {self.name} best score = {grid_search.best_score_}' ) ;
            self.logger.info(f'######### model {self.name} best params = {grid_search.best_params_}' ) ;

        self.writeModel( name=self.name )
        self.writeModelOpt( name=self.name )

    def testFeature(self):
        tmline = np.sort(self.ds.raw['Date'].unique());
        idx = int(tmline.shape[0] * 60 / 100) ;
        df = self.ds.getData(start=0, end=tmline[idx])
        idxx = int(tmline.shape[0] * 75 / 100) ;
        dft = self.ds.getData(start=tmline[idx+1], end=tmline[idxx])
        #
        score={}
        y = np.ravel(df['Target'].values)
        yt = np.ravel(dft['Target'].values)
        self.logger.info( f'#####################{self.name} start feature test')
        for col in df.columns:
            self.logger.info(f"Feature = {col}")
            if (not col.startswith('f_')):
                continue
            x = np.reshape(df[col].values, (-1,1))
            xt = np.reshape(dft[col].values, (-1,1))
            opt = {}
            opt['n_estimators'] = 100
            opt['max_depth'] = 4 
            opt['learning_rate'] = 0.1 
            opt['subsample'] = 0.8
            opt['colsample_bytree'] = 0.4 
            #opt['missing'] =-1, 
            opt['eval_metric']='mae'
            # USE CPU
            #nthread=4,
            #tree_method='hist' 
            # USE GPU
            #opt['tree_method‘】='gpu_hist' 
            self.model[self.name] = XGBRFRegressor(**opt) 
            try:
                self.fitModel(x,y)
                py = self.predict(x)
                val = self.calcPearson( y, py );
                self.logger.info(f'Feature {col} correlation = {val:<.5f}')
                pyt = self.predict(xt)
                val = self.calcPearson( yt, pyt );
                self.logger.info(f'Feature {col} test correlation = {val:<.5f}')
                score[col] = val
            except:
                traceback.print_exc()
                self.model[self.name] = None
        self.logger.info( f'#####################{self.name}')
        self.logger.info( f'{str(score)}')
        self.logger.info( f'#####################{self.name} end of feature test')

    ################
    def showPlot(self, wait = 5): # wait =  -1 wait forever
        if (wait > 0):
            plt.show(block=False)
            plt.pause(wait)
        else:
            plt.show(block=True)
    ################
    # plot heatmap
    def plotHeatmap(self, features=['Target','Close','Volume'], wait = 5 ): # wait =  -1 wait forever
        dt = self.ds.getData( )
        plt.figure(figsize=(12,9))
        sns.heatmap(dt[features].corr(), vmin=-1.0, vmax=1.0, annot=True, cmap='coolwarm', linewidths=0.1)
        self.showPlot(wait=wait)

    ################
    # plot correlation
    def plotCorrelation(self, wait = 5 ): # wait =  -1 wait forever
        check = pd.DataFrame()
        for i in self.investment_id:
            check[i] = self.ds.getData(i)['Target'].reset_index(drop=True) 
    
        plt.figure(figsize=(10,8))
        sns.heatmap(check.dropna().corr(), vmin=-1.0, vmax=1.0, annot=True, cmap='coolwarm', linewidths=0.1)
        self.showPlot(wait=wait)

# end class ModelGeneric
########################################

class ModelNNTest(ModelNN):
    def __init__(self, dataSource=None, name='model', opt=None):
        super(ModelNNTest, self).__init__(dataSource=dataSource, name=name, opt=opt)

# end class ModelGeneric
########################################


########################################
########################################
#
setupLogger( );

####################
globals()['USE_SELECT_FEATURE'] = False # load all data columns
globals()['USE_FEATURE_NORMALIZATION'] = False

ds = DataSource(dataPath=USE_DATA_PATH)
ds.printMemoryUssage('start')
#ds.readData() ;
ds.readData(fileName=USE_TRAIN_FILENAME)
ds.printMemoryUssage('readData')

if (USE_MODEL_LGB):
    lgm = ModelGenericTest(dataSource=ds,name=MODEL_NAME_LGB)
if (USE_MODEL_XGB):
    xgb = ModelGenericTest(dataSource=ds,name=MODEL_NAME_XGB)
if (USE_MODEL_NN):
    dnn = ModelNNTest(dataSource=ds,name=MODEL_NAME_DNN)

####################
DoTrainCV = True
if (DoTrainCV) :
    if (USE_MODEL_LGB):
        lgm.trainCV()
    if (USE_MODEL_XGB):
        xgb.trainCV()
    if (USE_MODEL_SVR):
        svr.trainCV()
    if (USE_MODEL_NN):
        dnn.train()
