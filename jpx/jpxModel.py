########################################
##
import pandas as pd
import numpy as np
import traceback
import os.path
import os
import pickle
import time
import datetime
from datetime import datetime
from timeit import default_timer as timer
import psutil
from scipy.stats import pearsonr

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
import tensorflow.keras.backend as K
import numpy.polynomial.hermite as Herm
import math
from tensorflow.python.ops import math_ops
from scipy import stats
import tensorflow_probability as tfp

from random import choices
import random
import keras_tuner as kt

from jpx import *
from jpxDataSource import *

########################################
# CONSTANT and setups
USE_ALL_TRAINING_SAMPLE = False

USE_InitModel = False

MODEL_NAME_TNN = 'TNN'
MODEL_NAME_DNN = 'DNN'
MODEL_NAME_LGB = 'LGB'
MODEL_NAME_XGB = 'XGBoost'
MODEL_NAME_SVR = 'SVR'
MODEL_NAME_RGF = 'RGF'

########################################
USETRAIN_SET_IN_PERCENT = 80
USETRAIN_FEATURE_CORR = 0.01

_SIZE_TRAIN = 300
_SIZE_TEST = 200
_GAP_TEST = 1

_TARGET_LIMIT = 1.0e8; #abs limit of the target value

########################################
# base model
class ModelBase(object):
    def __init__(self, dataSource=None, name='model', opt=None):
        # dataSource should always be provided even it is an empty one=
        self.logger = logging.getLogger('jpx.model')
        if (dataSource is None):
            dataSource = DataSource();
        self.name = name ;
        self.opt = opt;
        self.ds = dataSource ;

        self.model = {}
        self.modelType = {}
        self.modelWeight = {} ;
        self.modelOpt = {}

    ################
    # to be defined in each subclass
    def setDataSource(self, dataSource):
        self.ds = dataSource;
    
    ################
    # to be defined in each subclass
    def buildModel(self, mtype, X, Y, opt=None):
        raise NotImplementedError("buildModel error message")       

    ################
    # to be defined in each subclass
    def train(self, dataSource=None, opt=None):
        raise NotImplementedError("buildModel error message")       

    ################
    def getModelName(self, name=None ):
        if name is None:
            name='noname'
        cmn = f'model_{name}.p'
        return cmn

    ################
    def getModelOpt(self, name=None ):
        if name is None:
            name='noname'
        cmn = f'model_{name}_opt.p'
        return cmn

    ################
    def writeModel(self, name=None):
        for key in self.model:
            cmn = self.getModelName(name=name)
            self.logger.info(f'writeModel : {cmn}')
            with open(cmn, 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(self.model[key], f, pickle.HIGHEST_PROTOCOL) ;
            if (self.needScaler(name)) and (self.scaler != None):
                scmn = 'scaler_' + cmn
                self.logger.info(f'writeModel scaler : {scmn}')
                with open(scmn, 'wb') as f:
                    pickle.dump(self.scaler[key], f, pickle.HIGHEST_PROTOCOL) ;

    ################
    def readModel(self, name=None):
        for key in self.model:
            cmn = self.getModelName(name=name)
            if (os.path.exists(cmn)):
                self.logger.info(f'readModel : {cmn}')
                with open(cmn, 'rb') as f:
                    self.model[key] = pickle.load(f)
                if (self.needScaler(name)):
                    scmn = 'scaler_' + cmn
                    self.logger.info(f'readModel scaler: {scmn}')
                    with open(scmn, 'rb') as f:
                        self.scaler[key] = pickle.load(f)

    ################
    def writeModelOpt(self, name=None):
        for key in self.model:
            cmn = self.getModelOpt(name=name)
            self.logger.info(f'writeModel opt : {cmn}')
            with open(cmn, 'wb') as f:
                pickle.dump(self.modelOpt[key], f, pickle.HIGHEST_PROTOCOL) ;

    ################
    def readModelOpt(self, name=None):
        for key in self.model:
            cmn = self.getModelOpt(name=name)
            if (os.path.exists(cmn)):
                self.logger.info(f'readModel opt : {cmn}')
                with open(cmn, 'rb') as f:
                    self.modelOpt[key] = pickle.load(f)

    ################
    def calcPearson(self, preds, target):
        #
        target = np.nan_to_num(target)
        return pearsonr(target, np.nan_to_num(preds))[0]

    ################
    @classmethod
    def needScaler(cls, name ) :
        if (name.lower == MODEL_NAME_SVR.lower):
            return True ;
        else:
            return False; 

    ################
    def getFeature(self, start=None, end=None, iscategory=False ):
        x, y = self.ds.getFeature( start=start, end=end, iscategory=False )
        return x, y

## end class ModelBase
####################################

####################################
class ModelGeneric(ModelBase):
    def __init__(self, dataSource=None, name='model', opt=None):
        super(ModelGeneric, self).__init__(dataSource=dataSource, name=name, opt=opt)

    ################
    def buildModel(self, mtype=MODEL_NAME_LGB, X=None, Y=None, opt=None):
        if (mtype is None) or (mtype == MODEL_NAME_LGB) :
            if (opt is None):
                opt = {};
                opt['boosting_type'] = 'goss'
                opt['objective'] = 'regression_l2'
                opt['max_bin'] = 256
                opt['num_leaves'] = 253
                opt['learning_rate'] = 0.06
                opt['n_estimators'] = 2500
                opt['min_child_samples'] = 100
                opt['max_depth'] = 9
            model = LGBMRegressor(**opt)
        elif (mtype == MODEL_NAME_XGB):
            if (opt is None):
                opt = {};
                opt['booster'] = 'dart' #'dart' #gbtree gblinear
                opt['n_estimators'] = 2000
                opt['max_depth'] = 15 
                opt['learning_rate'] = 0.05
                opt['min_child_weight'] = 50
                opt['gamma'] = 100.0 
                opt['lambda'] = 100
                opt['top_k'] = 100
                opt['subsample'] = 0.9
                opt['colsample_bytree'] = 0.7
                #opt['missing'] =-1, 
                #opt['eval_metric']='rmse'
                # USE CPU
                #nthread=4,
                #tree_method='hist' 
                # USE GPU
                opt['tree_method']='gpu_hist' 
                opt['verbosity'] = 2
            model = XGBRFRegressor(**opt) 
        elif (mtype == MODEL_NAME_RGF):
            if (opt is None):
                opt = {};
                opt['loss'] = 'LS' #Expo/log
                opt['algorithm']='RGF' 
            model = RGFRegressor(**opt) 
        elif (mtype == MODEL_NAME_SVR):
            if (opt is None):
                opt = {}
            model = SVR(verbose=True, **opt )
        else:
            if (opt is None):
                opt = {}
            model = LGBMRegressor(**opt)
        self.logger.info(f'Model = {mtype} opt = {str(opt)}')
        return model

    ################
    def fitModel(self, name=MODEL_NAME_LGB, mtype=None, X=None, Y=None, init_model=None):
        if name is None:
            name = self.name;
        self.modelType[name] = mtype ;
        if (mtype is None) or (mtype == MODEL_NAME_LGB):
            self.model[name].fit(X, Y, eval_metric='regression_l1', init_model=init_model )
        elif (mtype == MODEL_NAME_XGB):
            self.model[name].fit(X.values, Y.values)
        elif (mtype == MODEL_NAME_RGF):
            self.model[name].fit(X.values, Y.values)
        elif (mtype == MODEL_NAME_SVR):
            self.model[name].fit(X, Y)
        else:
            self.model[name].fit(X, Y, eval_metric='regression_l2', init_model=init_model)

    ################
    def predictOneModel(self, name, x):
        try:
            if (self.modelType[name] == MODEL_NAME_XGB):
                y = self.model[name].predict( x.values )
            else:
                y = self.model[name].predict( x )
                #y[y < -_TARGET_LIMIT] = -_TARGET_LIMIT;
                #y[y >  _TARGET_LIMIT] =  _TARGET_LIMIT;
            return y ;
        except:
            self.logger.warning(f'predict error' ) ;
            traceback.print_exc()
            return 0

    def predict(self, x):
        yp = np.zeros((x.shape[0]))
        w = 0
        for name, model in self.model.items():
            y = self.predictOneModel(name, x)
            yp = yp + y ;
            if (self.modelWeight):
                w = w + self.modelWeight[name];
            else:
                w = w + 1;
        if (w != 0):
            yp = yp / w ;
        return yp ;

    ################
    # train on all data(split train/test)
    def train(self) :
        self.ds.printMemoryUssage('train')
        tmline = np.sort(self.ds.raw['Date'].unique());
        idx = int(tmline.shape[0] * USETRAIN_SET_IN_PERCENT / 100) ;
        st, ed = self.ds.getDataRange(0, tmline[idx])
        X, Y = self.getFeature() 
        if (USE_ALL_TRAINING_SAMPLE):
            x = X ;
            y = Y ;
        else:
            x = X.iloc[st:ed+1]
            y = Y.iloc[st:ed+1]
        self.ds.printMemoryUssage('train feature x y')
        #
        name = self.name
        params = self.opt
        self.logger.info(f"train {name} model")
        self.model[name] = self.buildModel(mtype=name, opt=params)
        try:
            self.fitModel(name=name, mtype=name, X=x, Y=y)
            self.writeModel(name)
            py = self.predict(x)
            val = self.calcPearson( y, py )
            self.logger.info(f'model {name} training correlation = {val:<.5f}')
            if (not USE_ALL_TRAINING_SAMPLE):
                xt = X.iloc[ed+1:]
                yt = Y.iloc[ed+1:]
                pyt = self.predict(xt)
                val = self.calcPearson( yt, pyt )
                self.logger.info(f'#################### model {name} training test correlation = {val:<.5f}')
        except:
            traceback.print_exc()
            self.logger.warning(f'train error' )
        gc.collect()

    def train2(self) :
        self.ds.printMemoryUssage('train')
        tmline = np.sort(self.ds.raw['Date'].unique());
        idx = int(tmline.shape[0] * USETRAIN_SET_IN_PERCENT / 100) ;
        st, ed = self.ds.getDataRange(0, tmline[idx])
        X, Y = self.getFeature() 
        if (USE_ALL_TRAINING_SAMPLE):
            x = X ;
            y = Y ;
        else:
            x = X.iloc[st:ed+1]
            y = Y.iloc[st:ed+1]
        self.ds.printMemoryUssage('train feature x y')

        sid = X['SecuritiesCode'].unique()

        #
        name = self.name
        self.logger.info(f"train {name} model")
        opt = {}
        opt['boosting_type'] = 'dart'
        opt['objective'] = 'regression_l2'
        opt['max_bin'] = 200
        opt['num_leaves'] = 31
        opt['learning_rate'] = 0.01
        opt['n_estimators'] = 1000
        opt['min_child_samples'] = 30
        opt['max_depth'] = 12
        xs = x[x['SecuritiesCode'] == sid[0]]
        ys = y[xs.index]
        xdefault = xs
        ydefault = ys

        try:
            for no, sname in enumerate(sid):
                self.logger.info(f'train no {no} name {sname}')
                self.model[sname] = self.buildModel(mtype=name, opt=opt)
                self.modelType[sname] = name
                xs = x[x['SecuritiesCode'] == sname]
                ys = y[xs.index]
                if (xs.shape[0] < 1):
                    xs = xdefault
                    ys = ydefault
                self.logger.info(f'features - {xs.shape[0]} x {xs.shape[1]}')
                self.fitModel(name=sname, mtype=name, X=xs, Y=ys)
                self.logger.info(f'end train no {no} name {sname}')
    
            py = self.predict2(x)
            val = self.calcPearson( y, py )
            self.logger.info(f'model {name} training correlation = {val:<.5f}')
            if (not USE_ALL_TRAINING_SAMPLE):
                xt = X.iloc[ed+1:]
                yt = Y.iloc[ed+1:]
                pyt = self.predict2(xt)
                val = self.calcPearson( yt, pyt )
                self.logger.info(f'#################### model {name} training test correlation2 = {val:<.5f}')
        except:
            traceback.print_exc()
            self.logger.warning(f'train error')
        gc.collect()

    ################
    # a simulated using rolling history
    def trainRolling(self) :
        # similate sumbission batch test
        tmline = np.sort(self.ds.raw['Date'].unique());
        dataSize = tmline[-1]
        trainEnd = int(dataSize * USETRAIN_SET_IN_PERCENT/100)
        cidx = _SIZE_TRAIN
        tstart = timer()
        subindex = int(0)
        while (cidx < trainEnd):
            st = max(int(cidx-_SIZE_TRAIN), 0)
            ed = min(int(cidx), trainEnd) ;
            tst = min(int(cidx+_GAP_TEST), dataSize) ;
            ted = min(int(tst+_SIZE_TEST), dataSize) ;
            if ((ed-st+1) < 100):
                cidx += _SIZE_TEST ;
                continue;
            
            x, y = self.getFeature(start=st, end=ed)
            xt, yt = self.getFeature(start=tst, end=ted)
            mtype = self.name
            name = f'{mtype}-{subindex:1d}'
            self.logger.info(f"trainRolling {mtype} model for {name}")
            self.model[name] = self.buildModel( mtype=mtype ) ;
            self.fitModel(name=name, mtype=mtype, X=x, Y=y)
            self.writeModel( name )
            try:
                pyt = self.predictOneModel(name, xt)
                corr = self.calcPearson( yt, pyt ) 
                self.logger.info( f'{name} model at {cidx} correlation = {corr}')
            except: 
                traceback.print_exc()
            cidx += _SIZE_TEST ;
            subindex = subindex + 1 ;
        xt, yt = self.getFeature(start=trainEnd+1, end=None)
        pyt = self.predict( xt )
        corrall = self.calcPearson( yt, pyt ) 

        tend = timer()
        self.logger.info( f'################### all {self.name} all models correlation = {corrall} in {tend-tstart:.2f}s')
        self.ds.gcCollect()
# end class ModelGeneric
########################################

########################################
class PatternEncoder(layers.Layer):
    def __init__(self, input_dim, pattern_dim, projection_dim):
        super(PatternEncoder, self).__init__()
        self.input_dim = input_dim
        self.pattern_dim = pattern_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=pattern_dim, output_dim=projection_dim
        )

    def call(self, pattern):
        positions = tf.range(start=0, limit=self.num_patterns, delta=1)
        encoded = self.projection(pattern) + self.position_embedding(positions)
        return encoded
    
class PerformancePlotCallback(keras.callbacks.Callback):
    def __init__(self, x_test, y_test):
        self.logger = logging.getLogger('jpx')
        self.x_test = x_test
        self.y_test = y_test
        
    def on_epoch_end(self, epoch, logs=None):
        pass

class ModelNN(ModelBase):
    def __init__(self, dataSource=None, name='model', opt=None):
        super(ModelNN, self).__init__(dataSource=dataSource, name=name, opt=opt)
        self.useClassification = False
        self.trainEpoch = USE_MODEL_NN_TRAIN_EPOCH
        self.customObjects = {'sharpe_loss': self.sharpe_loss,
                              'correlationLoss': self.correlationLoss,
                              'correlation': self.correlation,
                              }

    ################
    @staticmethod
    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    @staticmethod
    def correlationLoss(x,y, axis=-2):
        """Loss function that maximizes the pearson correlation coefficient between the predicted values and the labels,
        while trying to have the same mean and variance"""
        x = tf.convert_to_tensor(x)
        y = math_ops.cast(y, x.dtype)
        n = tf.cast(tf.shape(x)[axis], x.dtype)
        xsum = tf.reduce_sum(x, axis=axis)
        ysum = tf.reduce_sum(y, axis=axis)
        xmean = xsum / n
        ymean = ysum / n
        xsqsum = tf.reduce_sum( tf.math.squared_difference(x, xmean), axis=axis)
        ysqsum = tf.reduce_sum( tf.math.squared_difference(y, ymean), axis=axis)
        cov = tf.reduce_sum( (x - xmean) * (y - ymean), axis=axis)
        corr = tf.math.divide_no_nan(cov, tf.sqrt(xsqsum * ysqsum))
        return tf.convert_to_tensor( K.mean(tf.constant(1.0, dtype=x.dtype) - corr ) , dtype=tf.float32 )

    @staticmethod
    def correlation(x, y, axis=-2):
        """Metric returning the Pearson correlation coefficient of two tensors over some axis, default -2."""
        x = tf.convert_to_tensor(x)
        y = math_ops.cast(y, x.dtype)
        n = tf.cast(tf.shape(x)[axis], x.dtype)
        xsum = tf.reduce_sum(x, axis=axis)
        ysum = tf.reduce_sum(y, axis=axis)
        xmean = xsum / n
        ymean = ysum / n
        xvar = tf.reduce_sum( tf.math.squared_difference(x, xmean), axis=axis)
        yvar = tf.reduce_sum( tf.math.squared_difference(y, ymean), axis=axis)
        cov = tf.reduce_sum( (x - xmean) * (y - ymean), axis=axis)
        corr = tf.math.divide_no_nan(cov, tf.sqrt(xvar * yvar))
        return tf.constant(1.0, dtype=x.dtype) - corr

    @staticmethod
    def sharpe_loss(y,y_pred):
        #y_pred = tf.Variable(y_pred,dtype=tf.float32)
        port_ret = tf.reduce_sum(tf.multiply(y,y_pred),axis=1)
        s_ratio = tf.math.divide_no_nan(K.mean(port_ret), K.std(port_ret))
    
        return tf.math.exp(-s_ratio,  name='sharpe_loss')

    ################
    def setTrainEpoch(self, epoch=10):
        self.trainEpoch = epoch

    ################
    def buildModel(self, mtype=None, X=None, Y=None, opt=None):
        if (opt is None):
            opt = {};
        #
        if (mtype == MODEL_NAME_TNN):
            fsize = X.shape[1]-1
            f_input = tf.keras.Input((fsize,), dtype=tf.float32)

            pattern_dim = 32
            projection_dim = 32
            num_heads = 4
            transformer_units = [
                projection_dim * 2,
                projection_dim,
            ]  # Size of the transformer layers
            transformer_layers = 8
            mlp_head_units = [64, 32]  # Size of the dense layers of the final classifier

            iidfx = f_input ;

            # Encode pattern
            encoded_patches = PatternEncoder(fsize, pattern_dim, projection_dim)(iidfx)

            # Create multiple layers of the Transformer block.
            for _ in range(transformer_layers):
                # Layer normalization 1.
                x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
                # Create a multi-head attention layer.
                attention_output = layers.MultiHeadAttention(
                    num_heads=num_heads, key_dim=projection_dim, dropout=0.1
                )(x1, x1)
                # Skip connection 1.
                x2 = layers.Add()([attention_output, encoded_patches])
                # Layer normalization 2.
                x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
                # MLP.
                x3 = self.mlp(x=x3, hidden_units=transformer_units, dropout_rate=0.1)
                # Skip connection 2.
                encoded_patches = layers.Add()([x3, x2])

            # Create a [batch_size, projection_dim] tensor.
            representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            representation = layers.Flatten()(representation)

            representation = layers.Dropout(0.25)(representation)
            # Add MLP.
            fx = self.mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.25)
            output = layers.Dense(1, name='action')(fx)

            rmse = keras.metrics.RootMeanSquaredError(name="rmse")
            model = tf.keras.Model(inputs=[f_input, f_input], outputs=[output])
            model.compile(optimizer=tf.optimizers.Adam(0.001), loss='mae', metrics=['mse', "mae", "mape", rmse])
        else: # encoder/decoder
            # input feature
            """
            fsize = X.shape[-1]
            f_input = tf.keras.Input((fsize,), dtype=tf.float32)

            #weight = tf.Variable(tf.keras.backend.random_normal((fsize, 1), stddev=0.09, dtype=tf.float32))
            #var    = tf.Variable(tf.zeros((1,1), dtype=tf.float32))

            gaussian_noise = [0.02, 0.02]
            encoder_units = [896, 448, 50]
            dropout_rates = [0.40, 0.40, 0.40, 0.40]
            ae_units = [1004, 256]
            hidden_units = [38, 751, 1024, 58]
            lr = 1e-3 

            x0 = tf.keras.layers.BatchNormalization()(f_input)
            encoder = tf.keras.layers.GaussianNoise(gaussian_noise[0])(x0)

            encoder = tf.keras.layers.Dense(encoder_units[0])(encoder)
            #encoder = tf.keras.layers.Dense(encoder_units[1])(encoder)
            #encoder = tf.keras.layers.Dense(encoder_units[2])(encoder)
            #encoder = tf.keras.layers.BatchNormalization()(encoder)
            #encoder = tf.keras.layers.Activation('swish', name='encoder')(encoder)
    
            decoder = tf.keras.layers.Dropout(dropout_rates[0])(encoder)
            decoder = tf.keras.layers.Dense(fsize, name = 'decoder')(decoder)

            x_ae = tf.keras.layers.Dense(ae_units[0])(decoder)
            #x_ae = tf.keras.layers.BatchNormalization()(x_ae)
            #x_ae = tf.keras.layers.Activation('swish')(x_ae)
            x_ae = tf.keras.layers.Dropout(dropout_rates[1])(x_ae)

            if (self.useClassification):
                out_ae = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'ae_action')(x_ae)
            else:
                out_ae = tf.keras.layers.Dense(1, name = 'ae_action')(x_ae)
    
            ####
            x = tf.keras.layers.Concatenate()([x0, encoder])
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rates[2])(x)
    
            for i in range(len(hidden_units)):
                x = tf.keras.layers.Dense(hidden_units[i])(x)
                x = tf.keras.layers.BatchNormalization()(x)
                #x = tf.keras.layers.Activation('swish')(x)
                x = tf.keras.layers.Dropout(dropout_rates[3])(x)
        
            if (self.useClassification):
                out = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'action')(x)
            else:
                out = tf.keras.layers.Dense(1, name = 'action')(x)
    
            model = tf.keras.models.Model(inputs=f_input, outputs=[decoder, out_ae, out])

            #loss_out = tf.add(tf.matmul(f_input,weight), tf.math.reduce_sum(weight*var))
            #tf.compat.v1.losses.add_loss(loss_out)

            model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr),
                            #loss = {'decoder': [self.correlation, tf.keras.losses.MeanSquaredError(), tf.keras.metrics.CosineSimilarity(name='cosine')], 
                            #        'ae_action': tf.keras.losses.MeanSquaredError(),
                            #        'action': [self.sharpe_loss, self.correlation, tf.keras.losses.MeanSquaredError()], 
                            #       },
                            #metrics = {'decoder': [self.correlation, tf.keras.losses.MeanSquaredError(), tf.keras.metrics.CosineSimilarity(name='cosine')], 
                            #           'ae_action': tf.keras.metrics.MeanAbsoluteError(name = 'mae'), 
                            #           'action': [self.sharpe_loss, self.correlationLoss, tf.keras.metrics.MeanAbsoluteError(name = 'mae')], 
                            #          }, 
                            loss = {'decoder': tf.keras.losses.MeanSquaredError(), 
                                    'ae_action': tf.keras.losses.MeanSquaredError(),
                                    'action': tf.keras.losses.MeanSquaredError() 
                                    },
                            metrics = {'decoder': tf.keras.losses.MeanAbsoluteError(name = 'mae'), 
                                        'ae_action': tf.keras.metrics.MeanAbsoluteError(name = 'mae'), 
                                        'action': tf.keras.metrics.MeanAbsoluteError(name = 'mae')
                                      } 
                             )
            """
            fsize = X.shape[1]
            f_input = tf.keras.Input((fsize,), dtype=tf.float32)

            gaussian_noise = [0.02, 0.02]
            encoder_units = [896, 448, 50]
            decoder_units = [100, 448, 50]
            dropout_rates = [0.40, 0.40, 0.40, 0.40]
            ae_units = [1004, 256]
            hidden_units = [38, 751, 1024, 58]
            ls =  0 
            lr = 1e-3 

            x0 = tf.keras.layers.BatchNormalization()(f_input)
    
            encoder = tf.keras.layers.GaussianNoise(gaussian_noise[0])(x0)
            
            encoder = tf.keras.layers.Dense(encoder_units[0])(encoder)
            encoder = tf.keras.layers.BatchNormalization()(encoder)
            encoder = tf.keras.layers.Activation('swish')(encoder)
            encoder = tf.keras.layers.Dense(encoder_units[1])(encoder)
            encoder = tf.keras.layers.BatchNormalization()(encoder)
            encoder = tf.keras.layers.Activation('swish', name='encoder')(encoder)
    
            decoder = tf.keras.layers.Dropout(dropout_rates[0])(encoder)
            decoder = tf.keras.layers.Dense(decoder_units[0])(decoder)
            decoder = tf.keras.layers.BatchNormalization()(decoder)
            decoder = tf.keras.layers.Activation('swish',)(decoder)
            decoder = tf.keras.layers.Dense(fsize, name = 'decoder')(decoder)

            x_ae = tf.keras.layers.Dense(ae_units[0])(decoder)
            x_ae = tf.keras.layers.BatchNormalization()(x_ae)
            x_ae = tf.keras.layers.Activation('swish')(x_ae)
            x_ae = tf.keras.layers.Dropout(dropout_rates[1])(x_ae)

            if (self.useClassification):
                out_ae = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'ae_action')(x_ae)
            else:
                out_ae = tf.keras.layers.Dense(1, name = 'ae_action')(x_ae)
    
            ####
            x = tf.keras.layers.Concatenate()([x0, encoder])
            #x = x0
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(dropout_rates[2])(x)
    
            for i in range(len(hidden_units)):
                x = tf.keras.layers.Dense(hidden_units[i])(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Activation('swish')(x)
                x = tf.keras.layers.Dropout(dropout_rates[3])(x)
        
            if (self.useClassification):
                out = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'action')(x)
            else:
                out = tf.keras.layers.Dense(1, name = 'action')(x)
    
            model = tf.keras.models.Model(inputs = f_input, outputs = [decoder, out_ae, out])
            if (self.useClassification):
                model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr),
                              loss = {'decoder': tf.keras.losses.MeanSquaredError(), 
                                      'ae_action': tf.keras.losses.BinaryCrossentropy(label_smoothing = 1e-2),
                                      'action': tf.keras.losses.BinaryCrossentropy(label_smoothing = 1e-2), 
                                     },
                              metrics = {'decoder': tf.keras.metrics.MeanAbsoluteError(name = 'mae'), 
                                         'ae_action': tf.keras.metrics.AUC(name = 'AUC'), 
                                         'action': tf.keras.metrics.AUC(name = 'AUC'), 
                                        }, 
                             )
            else:
                model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr),
                              loss = {'decoder': tf.keras.losses.MeanSquaredError(), 
                                      'ae_action': tf.keras.losses.MeanSquaredError(),
                                      'action': tf.keras.losses.MeanSquaredError(), 
                                     },
                              metrics = {'decoder': tf.keras.metrics.MeanAbsoluteError(name = 'mae'), 
                                         'ae_action': tf.keras.metrics.MeanAbsoluteError(name = 'mae'), 
                                         'action': tf.keras.metrics.MeanAbsoluteError(name = 'mae'), 
                                        }, 
                             )
        ####

        model.summary()
        keras.utils.plot_model(model, show_shapes=True)
        self.logger.info(str(model.summary())) ;

        self.logger.info(f'Model = {mtype} opt = {str(opt)}')
        return model

    ################
    def makeDataset(self, X, Y=None):
        if (Y is None):
            Y = pd.DataFrame(data=np.zeros((X.shape[0],), dtype='float32'), columns=['target'])
        Z = (Y > 0) + 0
        return X.values, Y.values, Z.values
        
    def readModel(self, name=None):
        cmn = self.getModelName(name=name)
        self.model[name] = keras.models.load_model(cmn, custom_objects=self.customObjects)
        self.modelType[name] = self.name

    def fitModel(self, name=MODEL_NAME_DNN, mtype=None, X=None, Y=None, Xt=None, Yt=None):
        #
        x, y, z = self.makeDataset(X,Y)
        xt, yt, zt = self.makeDataset(Xt, Yt)
        cmn = self.getModelName(name=self.name)
        checkpoint = keras.callbacks.ModelCheckpoint(cmn, save_best_only=True)
        early_stop = keras.callbacks.EarlyStopping(patience=10)
        self.modelType[name] = mtype;
        if (mtype == MODEL_NAME_TNN):
            history = self.model[name].fit(x=x, y=y, batch_size=4000, epochs=self.trainEpoch, validation_data=(xt, yt), callbacks=[checkpoint, early_stop])
        else:
            if (self.useClassification):
                yabs = np.absolute(y);
                history = self.model[name].fit(x=x, y=[x, z, y], batch_size=4000, epochs=self.trainEpoch,
                                               validation_data=[xt, [xt, zt, yt]],
                                               sample_weight = yabs,
                                               callbacks=[checkpoint, early_stop])
            else:
                yabs = np.absolute(y);
                history = self.model[name].fit(x=x, y=[x, z, y], batch_size=4000, epochs=self.trainEpoch,
                                               validation_data=(xt, (xt, zt, yt)),
                                               #sample_weight = yabs,
                                               callbacks=[checkpoint, early_stop])

        #self.model[name] = keras.models.load_model(cmn, custom_objects=self.customObjects)
        #self.model[name] = keras.models.load_model(cmn, compile=False)
        self.model[name] = keras.models.load_model(cmn)
    
        pd.DataFrame(history.history, columns=["action_loss", "val_loss"]).plot()
        plt.title("loss")
        plt.show(block=False)
        time.sleep(5)

    ################
    def predictOneModel(self, name, x):
        try:
            xt, yt, zt = self.makeDataset(x)
            if (self.modelType[name] == MODEL_NAME_TNN):
                y = self.model[name].predict( xt ).ravel()
            else:
                y = self.model[name].predict( xt )
                y = y[2].ravel()
            return y
        except:
            self.logger.warning(f'predict error' ) ;
            traceback.print_exc()
            return 0

    def predict(self, x):
        yp = np.zeros((x.shape[0]))
        w = 0
        for name, model in self.model.items():
            y = self.predictOneModel( name, x)
            yp = yp + y ;
            if (self.modelWeight):
                w = w + self.modelWeight[name];
            else:
                w = w + 1;
        if (w != 0):
            yp = yp / w ;
        return yp ;

    ################
    # train on all data(split train/test)
    def train(self) :
        self.ds.printMemoryUssage('train')
        tmline = np.sort(self.ds.raw['Date'].unique());
        idx = int(tmline.shape[0] * USETRAIN_SET_IN_PERCENT / 100) ;
        st, ed = self.ds.getDataRange(0, tmline[idx])
        X, Y = self.getFeature() 
        x = X.iloc[st:ed] ;
        y = Y.iloc[st:ed] ;
        xt = X.iloc[ed:]
        yt = Y.iloc[ed:]
        self.ds.printMemoryUssage('train feature x y')
        #
        self.logger.info(f"Train {self.name} model")
        name = self.name;
        mtype = self.name ;
        self.model[name] = self.buildModel(mtype=mtype, X=X, Y=y)
        try:
            self.fitModel(name=name, mtype=mtype, X=x, Y=y, Xt=xt, Yt=yt)
            # mode saved
            py = self.predict(x)
            val = self.calcPearson( y, py );
            self.logger.info(f'model {self.name} training correlation = {val:<.5f}')
            if (not USE_ALL_TRAINING_SAMPLE):
                pyt = self.predict(xt)
                val = self.calcPearson( yt, pyt );
                self.logger.info(f'#################### model {self.name} training test correlation = {val:<.5f}')
        except: 
            traceback.print_exc()

    ################
    # a simulated using rolling history
    def trainRolling(self) :
        # similate sumbission batch test
        tmline = np.sort(self.ds.raw['Date'].unique());
        dataSize = tmline[-1]
        trainEnd = int(dataSize * USETRAIN_SET_IN_PERCENT/100)
        cidx = _SIZE_TRAIN
        tstart = timer()
        subindex = int(0)
        while (cidx < trainEnd):
            st = max(int(cidx-_SIZE_TRAIN), 0)
            ed = min(int(cidx), trainEnd) ;
            tst = min(int(cidx+_GAP_TEST), dataSize) ;
            ted = min(int(tst+_SIZE_TEST), dataSize) ;
            if ((ed-st+1) < 100):
                cidx += _SIZE_TEST ;
                continue;
            
            x, y = self.getFeature(start=st, end=ed)
            xt, yt = self.getFeature(start=tst, end=ted)
            mtype = self.name
            name = f'{mtype}-{subindex:1d}'
            self.logger.info(f"trainRolling {mtype} model for {name}")
            self.model[name] = self.buildModel( mtype=mtype, X=x, Y=y ) ;
            self.fitModel(name=name, mtype=mtype, X=x, Y=y, Xt=xt, Yt=yt)
            try:
                pyt = self.predictOneModel(name, xt)
                corr = self.calcPearson( yt, pyt ) 
                self.logger.info( f'{name} model at {cidx} correlation = {corr}')
            except: 
                traceback.print_exc()
            cidx += _SIZE_TEST ;
            subindex = subindex + 1 ;
        xt, yt = self.getFeature(start=trainEnd+1, end=None)
        pyt = self.predict( xt )
        corrall = self.calcPearson( yt, pyt ) 

        tend = timer()
        self.logger.info( f'################### all {self.name} all models correlation = {corrall} in {tend-tstart:.2f}s')
        self.ds.gcCollect()
# end class ModelNN
########################################

########################################
class ModelGenericNN(ModelGeneric):
    def __init__(self, dataSource=None, name='model', opt=None):
        super(ModelGenericNN, self).__init__(dataSource=dataSource, name=name, opt=opt)
        self.modelNN = keras.models.load_model('model_DNN.p')
        self.trainEpoch = 1

    ################
    def getFeature(self):
        X, Y = super(ModelGenericNN, self).getFeature()
        
        layer_output=self.modelNN.get_layer('encoder').output

        modelEncoder = tf.keras.models.Model(inputs=self.modelNN.input,outputs=layer_output)
        x = X;
        feature=modelEncoder.predict(x.values)
        xx = pd.DataFrame(feature) ;
        X = pd.concat([X,xx], axis=1)
        return xx,Y

    ################
    # train on all data(split train/test)
    ################
    # train on all data(split train/test)
    def train(self) :
        self.ds.printMemoryUssage('train')
        tmline = np.sort(self.ds.raw['Date'].unique());
        idx = int(tmline.shape[0] * USETRAIN_SET_IN_PERCENT / 100) ;
        st, ed = self.ds.getDataRange( 0, idx )
        X, Y = self.getFeature() 
        if (USE_ALL_TRAINING_SAMPLE):
            x = X ;
            y = Y ;
        else:
            x = X.iloc[st:ed+1]
            y = Y.iloc[st:ed+1]
        self.ds.printMemoryUssage('train feature x y')
        #
        name = self.name ;
        mtype = MODEL_NAME_LGB ;
        self.logger.info(f"train {self.name} model")
        self.model[name] = self.buildModel(mtype=mtype)
        try:
            self.fitModel(name=name, mtype=mtype, X=x, Y=y)
            self.writeModel(name)
            py = self.predict(x)
            val = self.calcPearson( y, py );
            self.logger.info(f'model {self.name} training correlation = {val:<.5f}')
            if (not USE_ALL_TRAINING_SAMPLE):
                xt = X.iloc[ed+1:]
                yt = Y.iloc[ed+1:]
                pyt = self.predict(xt)
                val = self.calcPearson ( yt, pyt );
                self.logger.info(f'#################### model {self.name} training test correlation = {val:<.5f}')
        except:
            traceback.print_exc()
            self.logger.warning(f'train error' ) ;
        gc.collect()

    ################
    # a simulated using rolling history
    def trainRolling(self) :
        # similate sumbission batch test
        tmline = np.sort(self.ds.raw['Date'].unique());
        dataSize = tmline[-1]
        trainEnd = int(dataSize * USETRAIN_SET_IN_PERCENT/100)
        cidx = _SIZE_TRAIN
        tstart = timer()
        subindex = int(0)
        while (cidx < trainEnd):
            st = max(int(cidx-_SIZE_TRAIN), 0)
            ed = min(int(cidx), trainEnd) ;
            tst = min(int(cidx+_GAP_TEST), dataSize) ;
            ted = min(int(tst+_SIZE_TEST), dataSize) ;
            if ((ed-st+1) < 100):
                cidx += _SIZE_TEST ;
                continue;
            
            x, y = self.getFeature(start=st, end=ed)
            xt, yt = self.getFeature(start=tst, end=ted)
            mtype = self.name
            name = f'{mtype}-{subindex:1d}'
            self.logger.info(f"trainRolling {mtype} model for {name}")
            self.model[name] = self.buildModel( mtype=mtype, X=x, Y=y ) ;
            self.fitModel(name=name, mtype=mtype, X=x, Y=y, Xt=xt, Yt=yt)
            try:
                pyt = self.predictOneModel(name, xt)
                corr = self.calcPearson( yt, pyt ) 
                self.logger.info( f'{name} model at {cidx} correlation = {corr}')
            except: 
                traceback.print_exc()
            cidx += _SIZE_TEST ;
            subindex = subindex + 1 ;
        xt, yt = self.getFeature(start=trainEnd+1, end=None)
        pyt = self.predict( xt )
        corrall = self.calcPearson( yt, pyt ) 

        tend = timer()
        self.logger.info( f'################### all {self.name} all models correlation = {corrall} in {tend-tstart:.2f}s')
        self.ds.gcCollect()
# end class ModelGeneric
########################################