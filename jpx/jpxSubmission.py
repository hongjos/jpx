########################################
##
import pandas as pd
import numpy as np
import logging
import traceback
import os.path
import os
import pickle
import time
import datetime
from datetime import datetime
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import psutil
import gc
from scipy.stats import pearsonr
import talib as ta

from rgf.sklearn import RGFRegressor
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from sklearn import linear_model
from xgboost import XGBRegressor
from xgboost.sklearn import XGBRFRegressor
from xgboost.sklearn import XGBRFClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

########################################
from jpx import *
from jpxDataSource import *
from jpxModel import *

########################################
# CONSTANT and setups
DO_SUBMISSION = 1 ; # = 0|None: submit real; 1 : local simulate; otherwise : DO_ nothing

####################################
class ModelSubmission(ModelBase):
    def __init__(self, dataSource=None, name='model', opt=None):
        super(ModelSubmission, self).__init__(dataSource=dataSource, name=name, opt=opt)
        # model
        if (USE_MODEL_LGB):
            self.modelLGB = ModelGeneric(dataSource=dataSource,name=MODEL_NAME_LGB, opt=opt) ;
        if (USE_MODEL_XGB):
            self.modelXGB = ModelGeneric(dataSource=dataSource,name=MODEL_NAME_XGB, opt=opt) ;
        if (USE_MODEL_RGF):
            self.modelRGF = ModelGeneric(dataSource=dataSource,name=MODEL_NAME_RGF, opt=opt) ;
        if (USE_MODEL_NN):
            self.modelNN = ModelNN(dataSource=dataSource,name=MODEL_NAME_DNN, opt=opt) ;
        if USE_MODEL_NNM2:
            self.modelNNM2 = ModelGenericNN(dataSource=dataSource,name=MODEL_NAME_LGB, opt=opt) ;
        self.loadModels() ;
        self.predTest = pd.DataFrame();
        self.predGT = pd.DataFrame();
        self.testCounter = 0 ;

    ################
    # load all models
    def loadModels(self):
        if (USE_MODEL_LGB):
            self.modelLGB.readModel(MODEL_NAME_LGB)
        if (USE_MODEL_XGB):
            self.modelXGB.readModel(MODEL_NAME_XGB)
        if (USE_MODEL_RGF):
            self.modelRGF.readModel(MODEL_NAME_RGF)
        if (USE_MODEL_NN):
            self.modelNN.readModel(MODEL_NAME_DNN)
        if USE_MODEL_NNM2:
            self.modelNNM2.readModel(MODEL_NAME_LGB)

    ################
    def appendDataTest(self, dftest):
        try:
            self.ds.appendDataTest(dftest)
        except:
            traceback.print_exc()

    def getFeature(self):
        try:
            tmline = np.sort(self.ds.raw['Date'].unique())
            dim = sum(self.ds.raw['Date'] == tmline[-1])
            st = tmline[0]
            ed = tmline[-1]
            df = self.ds.getData(start=st,end=ed)
            df = df.fillna(0)
            feature, target = self.ds.computeFeature(df)
            feature = feature.iloc[-dim:]
            return feature, target
        except:
            traceback.print_exc()

    ################
    def predict(self, investment_id=0):
        #import pdb; pdb.set_trace()
        x, _ = self.getFeature( ) ;
        yp = np.zeros((x.shape[0]), dtype=np.float32)
        wt = 0 ;
        if (USE_MODEL_LGB):
            yp += self.modelLGB.predict( x ) * 0.8
            wt += 0.8 ;
        if (USE_MODEL_XGB):
            yp = self.modelXGB.predict( x ) * 0.1
            wt += 0.1 ;
        if (USE_MODEL_NN):
            yp = self.modelNN.predict( x ) * 0.1
            wt += 0.1
        if (USE_MODEL_NNM2):
            yp = self.modelNNM2.predict( x ) * 0.8
            wt += 0.8
        ypred = yp / wt ;
        return ypred;

    ################
    def makeEnv(self):
        # read from file
        dss = DataSource(dataPath=USE_DATA_PATH) ;
        dss.readData('supplemental_files/stock_prices.csv') ;
        df = dss.raw ;
        groupValue = df['Date'].unique();
        self.predTest = pd.DataFrame();
        self.predGT = pd.DataFrame() ;
        # iterate through df
        for i, gp in enumerate(groupValue):
            adf = df[df['Date'] == gp].copy()
            dfpred = pd.DataFrame();
            dfpred['Date'] = adf['Date'] ;
            dfpred['RowId'] = adf['RowId']
            dfpred['Target'] = adf['Target'] ;
            dftest = adf;
            dftest.drop('Target', axis=1, inplace=True)
            yield dftest, dfpred

    ################
    def predictEnv(self, pred, predGT):
        self.predTest = pd.concat([self.predTest, pred])
        predGT = predGT.fillna(0)
        self.predGT = pd.concat([self.predGT, predGT])

    ################
    def calcResult(self):
        corr = self.calcPearson( self.predGT['Target'], self.predTest['Target'] )
        self.logger.info(f'correlation : {corr}')

    ################
    ## submission
    def submit(self, simulate=0):
        self.logger.info(f'Submission')
        if (simulate == 0) :
            ########
            logging.getLogger('jpx').warning(f'submission start')
            import jpx_tokyo_market_prediction
            env = jpx_tokyo_market_prediction.make_env()
            iter_test = env.iter_test()
            for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
                self.appendDataTest(prices)
                models_pred = self.predict()
                sample_prediction['prediction'] = models_pred
                sample_prediction = sample_prediction.sort_values(by='prediction', ascending=False)
                sample_prediction['Rank'] = np.arange(0, 2000)
                sample_prediction = sample_prediction.sort_values(by='SecuritiesCode', ascending=True)
                sample_prediction = sample_prediction.drop(columns=['prediction'])
                submission = sample_prediction[['Date', 'SecuritiesCode', 'Rank']]
                display(submission)
                env.predict(submission)
            logging.getLogger('jpx').warning(f'submission end')
            #import pdb; pdb.set_trace()
        elif simulate == 1: # simulate using supplemental_train.csv
            ########
            iter_test = self.makeEnv()
            tm = 0 ;
            tmNum = 0 ;
            for df_test, df_pred in iter_test:
                start = timer()
                #import pdb; pdb.set_trace()
                df_gt = pd.DataFrame();
                df_gt = df_pred.copy();
                self.appendDataTest( df_test ) 
                try:
                    df_pred['Target'] = self.predict( )
                except:
                    traceback.print_exc()
                self.predictEnv(df_pred, df_gt)
                end = timer()
                tm += (end-start)
                tmNum += 1 ;
                if (tmNum%10 == 9):
                    self.logger.info(f'average time = {tm/tmNum:.4f}')
                    self.calcResult();
            #import pdb; pdb.set_trace()
            self.calcResult();
            self.logger.info(f'#################### average time = {tm/tmNum:.4f}')
        else:
            ########
            self.logger.info(f'submit nothing')
        self.logger.info(f'End submission')
########################################

########################################
# submit
jpxMS = ModelSubmission()
jpxMS.submit(DO_SUBMISSION)