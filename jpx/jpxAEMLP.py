#@title GroupTimeSeriesSplit { display-mode: "form" }
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args

# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class GroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum size for a single training set.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import GroupTimeSeriesSplit
    >>> groups = np.array(['a', 'a', 'a', 'a', 'a', 'a',\
                           'b', 'b', 'b', 'b', 'b',\
                           'c', 'c', 'c', 'c',\
                           'd', 'd', 'd'])
    >>> gtss = GroupTimeSeriesSplit(n_splits=3)
    >>> for train_idx, test_idx in gtss.split(groups, groups=groups):
    ...     print("TRAIN:", train_idx, "TEST:", test_idx)
    ...     print("TRAIN GROUP:", groups[train_idx],\
                  "TEST GROUP:", groups[test_idx])
    TRAIN: [0, 1, 2, 3, 4, 5] TEST: [6, 7, 8, 9, 10]
    TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a']\
    TEST GROUP: ['b' 'b' 'b' 'b' 'b']
    TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] TEST: [11, 12, 13, 14]
    TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b']\
    TEST GROUP: ['c' 'c' 'c' 'c']
    TRAIN: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]\
    TEST: [15, 16, 17]
    TRAIN GROUP: ['a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b' 'c' 'c' 'c' 'c']\
    TEST GROUP: ['d' 'd' 'd']
    """
    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_size=None
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))
        group_test_size = n_groups // n_folds
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []
            for train_group_idx in unique_groups[:group_test_start]:
                train_array_tmp = group_dict[train_group_idx]
                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)
            train_end = train_array.size
            if self.max_train_size and self.max_train_size < train_end:
                train_array = train_array[train_end -
                                          self.max_train_size:train_end]
            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                              np.concatenate((test_array,
                                                              test_array_tmp)),
                                     axis=None), axis=None)
            yield [int(i) for i in train_array], [int(i) for i in test_array]
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args

# modified code for group gaps; source
# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]

                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)

            train_end = train_array.size

            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                              np.concatenate((test_array,
                                                              test_array_tmp)),
                                     axis=None), axis=None)

            test_array  = test_array[group_gap:]


            if self.verbose > 0:
                    pass

            yield [int(i) for i in train_array], [int(i) for i in test_array]
            
def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    cmap_cv = plt.cm.coolwarm
    jet     = plt.cm.get_cmap('jet', 256)
    seq     = np.linspace(0, 1, 256)
    _       = np.random.shuffle(seq)   # inplace
    cmap_data = ListedColormap(jet(seq))    
    for ii, (tr, tt) in enumerate(list(cv.split(X=X, y=y, groups=group))):
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0        
        ax.scatter(range(len(indices)), [ii + .5] * len(indices), c=indices, marker='_', lw=lw, cmap=cmap_cv, vmin=-.2, vmax=1.2)
    ax.scatter(range(len(X)), [ii + 1.5] * len(X), c=y, marker='_', lw=lw, cmap=plt.cm.Set3)
    ax.scatter(range(len(X)), [ii + 2.5] * len(X), c=group, marker='_', lw=lw, cmap=cmap_data)
    yticklabels = list(range(n_splits)) + ['target', 'day']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels, xlabel='Sample index', ylabel="CV iteration", ylim=[n_splits+2.2, -.2], xlim=[0, len(y)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax









#import gc, os, cudf
import gc, os
import talib as ta
import numpy as np
import pandas as pd
#import jpx_tokyo_market_prediction

from lightgbm import LGBMRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import RobustScaler

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

import warnings
warnings.filterwarnings("ignore")




import numpy.polynomial.hermite as Herm
import math
from tensorflow.python.ops import math_ops
from scipy import stats
import tensorflow_probability as tfp


from random import choices
import random
import keras_tuner as kt

device = "GPU"
if device == "GPU": print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
strategy = tf.distribute.get_strategy()
AUTO     = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync


def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)



SEED = 2025
set_all_seeds(SEED)

pathRoot = "E:/temp/Kaggle/"

path_s = pathRoot + "jpx-tokyo-stock-exchange-prediction/supplemental_files/"
path_t = pathRoot + "jpx-tokyo-stock-exchange-prediction/train_files/"
path_e = pathRoot + "jpx-tokyo-stock-exchange-prediction/example_test_files/"


prices1 = pd.read_csv(f"{path_s}stock_prices.csv") #2021
prices2 = pd.read_csv(f"{path_t}stock_prices.csv") #2017
prices3 = pd.read_csv(f"{path_t}secondary_stock_prices.csv")

prices1.shape , prices2.shape, prices3.shape


df = pd.concat([prices1,prices3, prices1])
df


def prep_prices(prices):
    prices.fillna(0,inplace=True)
    return prices


#simple units
hbar = 1.0
m    = 1.0
w    = 1.0

def hermite(x, n):
    xi             = np.sqrt(m*w/hbar)*x
    herm_coeffs    = np.zeros(n+1)
    herm_coeffs[n] = 1
    return Herm.hermval(xi, herm_coeffs)

def stationary_state(x,n):
    xi        = np.sqrt(m*w/hbar)*x
    prefactor = 1.0/math.sqrt(2.0**n * math.factorial(n)) * (m*w/(np.pi*hbar))**(0.25)
    psi       = prefactor * np.exp(- xi**2 / 2) * hermite(x,n)
    return psi


NDAYS = 180
lastdays = df[df["Date"]>=df["Date"].iat[-2000*NDAYS]].reset_index(drop=True)


lastdays = pd.DataFrame(df.groupby("SecuritiesCode").Target.mean())
def get_avg(_id_):
    return lastdays.loc[_id_]


def features(df, tr=True):
    df = prep_prices(df)
    
    '''
    # Encrypt Strategy #simple units
    seed_n = 27
    
    if tr:
        scale_features = df.columns.drop(['SecuritiesCode','Date','Target'])    
        df[scale_features] = RobustScaler().fit_transform(df[scale_features]) 
    if not tr:
        scale_features = df.columns.drop(['SecuritiesCode','Date'])    
        df[scale_features] = RobustScaler().fit_transform(df[scale_features]) 
    
    df['Close']    = stationary_state(df['Close'], seed_n) 
    df['Open']     = stationary_state(df['Open'], seed_n) 
    df['Low']      = stationary_state(df['Low'], seed_n)
    df['High']     = stationary_state(df['High'], seed_n)
    df['Volume']   = stationary_state(df['Volume'], seed_n) 
    '''
    
    
    df['upper_Shadow']   = df['High'] - np.maximum(df['Close'], df['Open'])
    df['lower_Shadow']   = np.minimum(df['Close'], df['Open']) - df['Low'] 

    # The Golden Ratio Multiplier 
    df['GRM_0']    = (ta.MA(df['Close'], timeperiod=350, matype=0)) 
    df['GRM_1']    = (ta.MA(df['Close'], timeperiod=350, matype=0))*1.6  
    df['GRM_2']    = (ta.MA(df['Close'], timeperiod=350, matype=0))*2
    df['GRM_3']    = (ta.MA(df['Close'], timeperiod=350, matype=0))*3

    df['Pi_Cycle'] = ta.MA(df['Close'], timeperiod=111, matype=0) 
    
    # Momentum
    df['RSI_14'] = ta.RSI(df['Close'], timeperiod=14)
    df['RSI_24'] = ta.RSI(df['Close'], timeperiod=24)
    
    df['RSI1']   = df['RSI_14'].shift(-1) 
    df['RSI4']   = df['RSI_14'].shift(-4) 
    df['RSI7']   = df['RSI_14'].shift(-7) 
    df['RSI10']  = df['RSI_14'].shift(-10) 
    df['RSI13']  = df['RSI_14'].shift(-13) 
    df['RSI16']  = df['RSI_14'].shift(-16) 
    
    df['MACD_12'], df['macdsignal_12'], df['MACD_HIST_12'] = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9) 
    df['MACD_48'], df['macdsignal_48'], df['MACD_HIST_48'] = ta.MACD(df['Close'], fastperiod=48, slowperiod=104, signalperiod=36)
    
    df['macdsignal1'] = df['macdsignal_12'].shift(-1)
    df['macdsignal4'] = df['macdsignal_12'].shift(-4)
    df['macdsignal7'] = df['macdsignal_12'].shift(-7)
    df['MACD_HIST1']  = df['MACD_HIST_12'].shift(-1) 
    df['MACD_HIST4']  = df['MACD_HIST_12'].shift(-4) 
    df['MACD_HIST7']  = df['MACD_HIST_12'].shift(-7) 
    df['ROCP']     = ta.ROCP(df['Open'])
    df['momentam'] = ta.MOM(df['Open'])
    df['CMO']      = ta.CMO(df['Open']) 
    df['PPO']      = ta.PPO(df['Open'])
    df['SAR']       = ta.SAR(df['High'], df['Low'], acceleration=0, maximum=0) 
    df['DI_minus']  = ta.MINUS_DI(df['High'], df['Low'],np.array(df.loc[:, 'Close']), timeperiod=14) 
    df['DI_minus1'] = df['DI_minus'].shift(-1) 
    df['DI_minus4'] = df['DI_minus'].shift(-4) 
    df['DI_minus7'] = df['DI_minus'].shift(-7)  
    df['adx']    = ta.ADX(df['High'], df['Low'],np.array(df.loc[:, 'Close']),timeperiod=14) 
    df['adx1']   = df['adx'].shift(-1) 
    df['adx4']   = df['adx'].shift(-4) 
    df['adx+1']  = df['adx'].shift(1) 
    df['adx7']   = df['adx'].shift(-7)
    df['DI_plus']   = ta.PLUS_DI(df['High'], df['Low'],np.array(df.loc[:, 'Close']), timeperiod=14) 
    df['DI_plus1']  = df['DI_plus'].shift(-1) 
    df['DI_plus4']  = df['DI_plus'].shift(-4) 
    df['DI_plus7']  = df['DI_plus'].shift(-7) 
    df['DI_plus10'] = df['DI_plus'].shift(-10)
    df['APO']      = ta.APO(df['Open'])
    df['APO1']     = df['APO'].shift(-1)
    df['APO4']     = df['APO'].shift(-4)
    df['APO7']     = df['APO'].shift(-7)
    df['ROCR100']  = ta.AD(df['High'], df['Low'], df['Close'], df['Volume'])
    df['OBV']      = ta.OBV(df['Close'], df['Volume'])
    df['ADOSC']    = ta.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'], fastperiod=3, slowperiod=10)
    df['ATR']    = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['NATR']   = ta.NATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['Variance'] = ta.VAR(df['Close'], timeperiod=5, nbdev=1)
    df['CORREL']   = ta.CORREL(df['High'], df['Low'], timeperiod=15)
    df['TSF']      = ta.TSF(df['Close'], timeperiod=14) 
    df['TSF-14']   = ta.TSF(df['Close'], timeperiod=14).shift(-14)
    df['TSF-7']    = ta.TSF(df['Close'], timeperiod=14).shift(-7)
    df['ATR']    = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['NATR']   = ta.NATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    
    df['s_avg']      = df['SecuritiesCode'].apply(get_avg)
    df['qhm_115v']   = stationary_state(df['s_avg'], 115) 
    
    df['Date']      = pd.to_datetime(df['Date'])
    df['weekday']   = df['Date'].dt.weekday+1
    df['Monday']    = np.where(df['weekday']==1,1,0)
    df['Tuesday']   = np.where(df['weekday']==2,1,0)
    df['Wednesday'] = np.where(df['weekday']==3,1,0)
    df['Thursday']  = np.where(df['weekday']==4,1,0)
    df['Friday']    = np.where(df['weekday']==5,1,0)
    
    ema_set = [3,5,8,12,15,26,30,35,40,45,50,60, 100,200]
    # EMA
    for i in range(len(ema_set)):
        sma = df['Close'].rolling(ema_set[i]).mean()
        ema = sma.ewm(span=ema_set[i], adjust=False).mean()
        df["EMA_%d"%(ema_set[i])] = ema
        df = prep_prices(df)
    
    if tr:
        df = df.drop(['RowId','Target','AdjustmentFactor','ExpectedDividend','SupervisionFlag','Date'],axis=1)
    return df 




X = features(df)
y = df.Target
groups = pd.factorize(pd.to_datetime(df['Date']).dt.strftime('%d').astype(str) + '_' + pd.to_datetime(df['Date']).dt.strftime('%m').astype(str) + '_' +pd.to_datetime(df['Date']).dt.strftime('%Y').astype(str))



# CV PARAMS
FOLDS                = 3
GROUP_GAP            = 14
MAX_TEST_GROUP_SIZE  = 180  
MAX_TRAIN_GROUP_SIZE = 485

# USE VERBOSE=0 for silent, VERBOSE=1 for interactive, VERBOSE=2 for commit
VERBOSE = 2


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
fig, ax = plt.subplots(figsize = (12, 6))
cv = PurgedGroupTimeSeriesSplit(n_splits = FOLDS, group_gap = GROUP_GAP, max_train_group_size=MAX_TRAIN_GROUP_SIZE, max_test_group_size=MAX_TEST_GROUP_SIZE)
plot_cv_indices(cv, X, y, groups[0], ax, FOLDS, lw=20)


X.shape, y.shape


from tensorflow.python.keras import backend as K
def e_swish(beta=0.25):
    def beta_swish(x): return x*K.sigmoid(x)*(1+beta)
    return beta_swish









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
    corr = cov / tf.sqrt(xsqsum * ysqsum)
    return tf.convert_to_tensor( K.mean(tf.constant(1.0, dtype=x.dtype) - corr ) , dtype=tf.float32 )



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
    corr = cov / tf.sqrt(xvar * yvar)
    return tf.constant(1.0, dtype=x.dtype) - corr



def sharpe_loss(y,y_pred):
    y_pred = tf.Variable(y_pred,dtype=tf.float64)
    port_ret = tf.reduce_sum(tf.multiply(y,y_pred),axis=1)
    s_ratio = K.mean(port_ret)/K.std(port_ret)
    
    return tf.math.exp(-s_ratio,  name='sharpe_loss')



def build_model(hp, dim = 128, fold=0):

    features_inputs = tf.keras.layers.Input(shape = (dim, ))
    x0      =  tf.keras.layers.BatchNormalization()(features_inputs)
    
    weight = tf.Variable(tf.keras.backend.random_normal((dim, 1), stddev=hp.Float(f'weight_{fold}',1e-10, 0.09), dtype=tf.float32))
    var    = tf.Variable(tf.zeros((1,1), dtype=tf.float32))
   
    encoder = tf.keras.layers.GaussianNoise(0.4)(x0)
    encoder = tf.keras.layers.Dense(hp.Int(f'layers{fold}_en0',32, 1024))(encoder)
    encoder = tf.keras.layers.Dense(hp.Int(f'layers{fold}_en1',32, 1024))(encoder)
    encoder = tf.keras.layers.Dense(hp.Int(f'layers{fold}_en2',32, 1024))(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Activation(e_swish(beta=hp.Float(f'e{fold}_en0',0.001, 1 )))(encoder)
    
    decoder = tf.keras.layers.Dropout(hp.Float(f'dropout{fold}_de0',0.001, 0.8))(encoder)
    decoder = tf.keras.layers.Dense(hp.Int(f'layers{fold}_de0',32, 1024), name='decoder')(decoder)
    
    x_ae = tf.keras.layers.Dense(hp.Int(f'layers{fold}_ae0',32, 1024))(decoder)
    x_ae = tf.keras.layers.BatchNormalization()(x_ae)
    x_ae = tf.keras.layers.Activation(e_swish(beta=hp.Float(f'e{fold}_ae0',0.001, 1 )))(x_ae)
    x_ae = tf.keras.layers.Dropout(hp.Float(f'dropout{fold}_ae0',0.001, 0.8))(x_ae) 
    
    feature_x = tf.keras.layers.Concatenate()([x0, encoder])
    feature_x = tf.keras.layers.BatchNormalization()(feature_x)
    feature_x = tf.keras.layers.Dense(hp.Int(f'layers{fold}_fx0',32, 1024))(feature_x)
    feature_x = tf.keras.layers.Activation(e_swish(beta=hp.Float(f'e_fx0',0.001, 1 )))(feature_x)
    feature_x = tf.keras.layers.Dropout(hp.Float(f'dropout{fold}_fx0',0.001, 0.8))(feature_x)

    x = layers.Dense(hp.Int(f'layers{fold}_x0',32, 1024), activation= e_swish(beta=hp.Float(f'e{fold}_x0',0.001, 1 )), kernel_regularizer="l2")(feature_x)
    x = layers.Dense(hp.Int(f'layers{fold}_x1',32, 1024), activation= e_swish(beta=hp.Float(f'e{fold}_x1',0.001, 1 )), kernel_regularizer="l2")(x)
    x = layers.Dense(hp.Int(f'layers{fold}_x2',32, 1024), activation= e_swish(beta=hp.Float(f'e{fold}_x2',0.001, 1 )), kernel_regularizer="l2")(x)
    x = layers.Dense(hp.Int(f'layers{fold}_x3',32, 1024), activation= e_swish(beta=hp.Float(f'e{fold}_x3',0.001, 1 )), kernel_regularizer="l2")(x)
    x = tf.keras.layers.Dropout(hp.Float(f'dropout{fold}_x0',0.001, 0.8))(x)

    mlp_out = layers.Dense(1, name ='mlp_out')(x)

    model  = tf.keras.Model(inputs=[features_inputs], outputs=[decoder, mlp_out])
    
    loss_out = tf.add(tf.matmul(features_inputs,weight), tf.math.reduce_sum(weight*var))
    tf.compat.v1.losses.add_loss(loss_out)
  
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Float(f'lr_adam{fold}',1e-3, 1e-5)),
                  loss = {'decoder': [tf.keras.losses.CosineSimilarity(axis=-2), 
                                      tf.keras.losses.MeanSquaredError(), 
                                      correlationLoss],         
                          
                          'mlp_out' : [sharpe_loss],
                         },
                  metrics = {'decoder': [tf.keras.metrics.CosineSimilarity(name='cosine'),
                                         tf.keras.metrics.MeanAbsoluteError(name="mae"), 
                                         correlation, 
                                         tf.keras.metrics.RootMeanSquaredError(name='rmse')], 
                             
                             'mlp_out' : [tf.keras.metrics.CosineSimilarity(name='cosine'),
                                          tf.keras.metrics.MeanAbsoluteError(name="mae"), 
                                          correlation, 
                                          tf.keras.metrics.RootMeanSquaredError(name='rmse')],
                            },
                 ) 
    return model


hp = pd.read_pickle(pathRoot + f'jpx-tokyo-stock-exchange-prediction/hp-jpx-aemlp/best_hp_ae_jpx_3gkf.pkl')


tf.keras.utils.plot_model(build_model(hp, fold=0), show_shapes=True, expand_nested=True, show_dtype=True)


batch_size = [4096*4,4096,4096*8]

gkf = PurgedGroupTimeSeriesSplit(n_splits = FOLDS, 
                                 group_gap = GROUP_GAP, 
                                 max_train_group_size = MAX_TRAIN_GROUP_SIZE, 
                                 max_test_group_size  = MAX_TEST_GROUP_SIZE).split(X, y, groups[0])
models = []
for fold, (train_idx, val_idx) in enumerate(list(gkf)):
    x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    print(f'>>> AEMLP_FOLD:{fold}')
    K.clear_session()
    with strategy.scope(): model = build_model(hp, dim = x_train.shape[1], fold=fold)
    model_save = tf.keras.callbacks.ModelCheckpoint('./fold-%i.hdf5' %(fold), 
                                                         monitor = 'val_mlp_out_rmse', verbose = 0, 
                                                         save_best_only = True, save_weights_only = True,
                                                         mode = 'min', save_freq = 'epoch')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mlp_out_rmse', patience=15, mode='min', restore_best_weights=True)
    history = model.fit(x_train, y_train ,
                        epochs          = 200, 
                        callbacks       = [model_save, early_stop], 
                        validation_data = (x_val, y_val), 
                        batch_size      = batch_size[fold],
                        verbose         = 2) 
    print('='*96)
    models.append(model)
    gc.collect()



'''
models = []
for fold, (train_idx, val_idx) in enumerate(gkf):
    x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    print(f'>>> LOAD_AEMLP_FOLD:{fold}')
    K.clear_session()
    with strategy.scope(): model = build_model(hp, dim = x_train.shape[1], fold=fold)
    model.load_weights(f'../input/hp-jpx-aemlp-3gkf-weight/fold-{fold}.hdf5') 
    models.append(model)
    gc.collect()
'''


dfx = features(prices1)
dfy = prices1.Target


ap = [0.10,0.10,0.80]
model_x = list()
for i in range(FOLDS):
    prediction_x = models[i].predict(dfx)[-1] * ap[i]
    model_x.append(prediction_x)
model_x = np.mean(model_x, axis = 0)
dfx['pre'] = model_x


dfx.shape, dfy.shape


from scipy import stats
pearson_score = stats.pearsonr(dfx['pre'], dfy)[0]
print('Pearson:', pearson_score)

feats = X.columns














