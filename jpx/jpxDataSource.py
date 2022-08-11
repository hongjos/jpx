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
import plotly.graph_objects as go
import seaborn as sns
import psutil
import gc
from scipy.stats import pearsonr
import talib as ta

USE_FEATURE_SELECT = True
USE_FEATURE_NORMALIZATION = False
USE_FEATURE_NORMALIZATION_STD_RANGE = 4

USE_FEATURE_EXTRA = 0

USE_DATA_PATH = "C:/Temp/Kaggle/jpx-tokyo-stock-exchange-prediction/"
if ('DO_SUBMISSION' in globals()) and (DO_SUBMISSION > 0):
    USE_DATA_PATH = "/kaggle/input/jpx-tokyo-stock-exchange-prediction/"
USE_TRAIN_FILENAME = ['train_files/stock_prices.csv',\
                      'train_files/secondary_stock_prices.csv',\
                      'supplemental_files/stock_prices.csv']
USE_TRAIN_FILENAME2 = ['train_files/stock_prices.csv']

_MEM_VERBOSITY = 1

########################################
# log
def setupLogger(name='jpx.log')->None:
    logger = logging.getLogger('jpx')
    if ('DO_SUBMISSION' in globals() and DO_SUBMISSION == 0):
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    h = logging.FileHandler(name, mode='a')
    h.setLevel(logging.INFO)
    fh = logging.StreamHandler()
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    #formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    #formatter = logging.Formatter('%(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    formatterDetail = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    h.setFormatter(formatter)
    fh.setFormatter(formatter)
    ch.setFormatter(formatterDetail)
    # add the handlers to the logger
    logger.addHandler(h)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f'') # an empty log item
    logger.info(f'')

########################################
# data source class
class DataSource(object):
    ################
    ################
    def __init__(self, dataPath=None):
        self.logger = logging.getLogger('jpx.source')
        self.dataPath=dataPath;
        if (self.dataPath is None):
            self.dataPath = "/kaggle/input/jpx-tokyo-stock-exchange-prediction/"
        self.raw = pd.DataFrame();
        self.sid = []
        self.minSequenceLength = 60

    ################
    # setPath - set dataPath
    def setPath(self, dataPath ):
        self.dataPath = dataPath;

    ################
    # setup a dataset of empty
    def setupDataEmpty(self):
        self.raw = pd.DataFrame();

    ################
    # setup a dataset of empty
    def gcCollect(self, wait=5):
        gc.collect()
        time.sleep(wait)

    ################
    def printMemoryUssage(self, strInfo='', verbose=0):
        if (_MEM_VERBOSITY >= verbose):
            rmem = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
            self.logger.info( f'{strInfo} resident memory = {rmem:.2f}G')

    ################
    def reduceMemory(self, df):
        before = df.memory_usage().sum()  
        for col in df.columns:        
            dtype = df[col].dtype
            if dtype == 'float64':
                c_min = df[col].min()
                c_max = df[col].max()        
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #
        df['SecuritiesCode'] = df['SecuritiesCode'].astype('int32')
        after = df.memory_usage().sum()
        self.logger.info( f'Memory taken before transformation : {before:d}' )
        self.logger.info( f'Memory taken after transformation : {after:d}')
        self.logger.info( f'Memory taken reduced by : {(before - after) * 100/ before:.1}%%')
        return df

    ################
    # read a dataset from training, clean data, split into asset df
    def readData(self, fileName='train_files/stock_prices.csv', dataPath=None):
        if (not dataPath is None):
            self.setPath(dataPath)
        # initialize to empty df
        self.setupDataEmpty()
        # read from file
        if (not isinstance(fileName, list)):
            fileList = [fileName]
        else:
            fileList = fileName

        dflist = []
        for name in fileList:
            self.logger.info(f'Load data {name}')
            if (name.lower().endswith('.gzip')) or (name.lower().endswith('.parquet')):
                dflist.append(pd.read_parquet(self.dataPath + name))
            else:
                dflist.append(pd.read_csv(self.dataPath + name))
        self.raw = pd.concat(dflist)
        self.raw.drop_duplicates(['SecuritiesCode', 'Date'])
        self.raw = self.reduceMemory( self.raw )
        # preprocess data
        self.gcCollect()
        self.printMemoryUssage( 'preprocessData')
        self.raw = self.preprocessData(self.raw)
        mems = self.raw.memory_usage().sum()
        self.logger.info( f'Memory size = {mems/1000_0000:.0f}M' )
        self.printMemoryUssage( 'readData Done')
        self.logger.info(f'Load data Done')

    ################
    def preprocessData(self, df):
        return df

    ################
    def getSID(self):
        self.sid = self.raw['SecuritiesCode'].unique()
        return self.sid

    ################
    # add test data to data
    def appendDataTest(self, dftest):
        try:
            df = dftest.copy()
            df = self.preprocessData(df)
            self.raw = pd.concat([self.raw, df])
        except:
            rowId = dftest['RowId'].iloc[0]
            self.logger.warning(f'Append data exception {rowId}')
            traceback.print_exc()

    ################
    # getDataRange
    def getDataRange(self, start=None, end=None):
        st = 0
        if (not start is None):
            if (isinstance(start, str)):
                traw = self.raw['Date'];
                idx = (traw >= start);
                start = 0; # is ts not found, use begin
                if (traw.shape[0] > 0):
                    idx = np.arange(idx.size)[idx] ;
                    if (idx.size > 0):
                        st = idx[0]
            else: # treated as index value
                rows = self.raw.shape[0]
                st = min(max(start, 0), rows)

        en = self.raw.shape[0]
        if (not end is None):
            if (isinstance(end, str)):
                traw = self.raw['Date'];
                idx = (traw <= end)
                if (traw.shape[0] > 0):
                    idx = np.arange(idx.size)[idx] ;
                    if (idx.size > 0):
                        en = idx[-1]+1
            else:
                rows = self.raw.shape[0]
                en = min(max(end, 0), rows)
        return st, en

    ################
    # getData
    def getData(self, start=None, end=None, minSeriesLength=0 ):
        st, ed = self.getDataRange( start, end )
        if (st == 0) and (ed == self.raw.shape[0]):
            return self.raw
        else:
            return self.raw.iloc[st:ed]

    ################
    # get sequence
    def getDataSequence(self, sid):
        return self.raw[self.raw['SecuritiesCode'] == sid]

    def updateDataSequence(self, sid, column, value):
        self.raw.loc[self.raw['SecuritiesCode'] == sid, column] = value


    ################
    # computeFeature
    def computeFeature(self, dfraw, iscategory=False):
        # feature dimension
        
        pd.options.mode.chained_assignment = None  # default='warn'

        df = dfraw.copy()
        nrow = df.shape[0] ;
        feature = df

        df.fillna(0, inplace=True)
        df.sort_values('Date', inplace=True)
        # assuming data in df is date increasing -> the most recent is the last

        df["Volume_ratio"] = df["Volume"]/df.groupby("SecuritiesCode")["Volume"].rolling(window=15, min_periods=1).mean().reset_index("SecuritiesCode",drop=True)
        
        df["Open"] = df.groupby("SecuritiesCode").apply(lambda d:d["Open"]/d["AdjustmentFactor"].cumprod().shift().fillna(1)).reset_index("SecuritiesCode",drop=True)
        df["High"] = df.groupby("SecuritiesCode").apply(lambda d:d["High"]/d["AdjustmentFactor"].cumprod().shift().fillna(1)).reset_index("SecuritiesCode",drop=True)
        df["Low"] = df.groupby("SecuritiesCode").apply(lambda d:d["Low"]/d["AdjustmentFactor"].cumprod().shift().fillna(1)).reset_index("SecuritiesCode",drop=True)
        df["Close"] = df.groupby("SecuritiesCode").apply(lambda d:d["Close"]/d["AdjustmentFactor"].cumprod().shift().fillna(1)).reset_index("SecuritiesCode",drop=True)
        
        df['upper_Shadow']   = (df['High'] - np.maximum(df['Close'], df['Open'])) / df['Close']
        df['lower_Shadow']   = (np.minimum(df['Close'], df['Open']) - df['Low']) / df['Close']

        
        self.computeFeatureSequence( df )

        df.fillna(0, inplace=True)


        if ('Date' in feature.columns):
            feature = feature.drop( ['RowId', 'Date'], axis=1)
        else:
            feature = feature.drop( ['RowId'], axis=1)
        if ('SecuritiesCode' in feature.columns):
            feature = feature.drop( ['SecuritiesCode'], axis=1)
        if ('SupervisionFlag' in feature.columns):
            feature = feature.drop( ['SupervisionFlag'], axis=1)
        if ('AdjustmentFactor' in feature.columns):
            feature = feature.drop( ['AdjustmentFactor'], axis=1)

        self.printMemoryUssage('computeFeature', verbose=1)
        if ('Target' in feature.columns):
            target = feature['Target']
            feature = feature.drop( ['Target'], axis=1)
            target = target.fillna(0)
        else:
            target = pd.DataFrame(data=np.zeros((nrow,)), columns=['Target'])
        if (iscategory == True): # class feature
            target[target > 0] = 1 ;
            target[target < 0] = 0 ;
            target = target.astype(int);
        feature = feature.fillna(0)
        #
        return feature, target

    @staticmethod
    def MA(series, window=25):
        return series.rolling(window, min_periods=1).mean()

    @staticmethod
    def DMA(series, window=25):
        return series/DataSource.MA(series, window) - 1

    @staticmethod
    def divergence(series, window=25):
        std = series.rolling(window,min_periods=1).std()
        mean = series.rolling(window,min_periods=1).mean()
        return (series-mean) / std    

    @staticmethod
    def rsi(series, n=14):
        return (series - series.shift(1)).rolling(n).apply(lambda s:s[s>0].sum()/abs(s).sum())

    @staticmethod
    def stochastic(series, k=14, n=3, m=3):
        _min = series.rolling(k).min()
        _max = series.rolling(k).max()
        _k = (series - _min)/(_max - _min)
        _d1 = _k.rolling(n).mean()
        _d2 = _d1.rolling(m).mean()
        return pd.DataFrame({
                        "%K":_k,
                        "FAST-%D":_d1,
                        "SLOW-%D":_d2,
                        },index=series.index)
        # return _k, _d1, _d2

    @staticmethod
    def createMAFeature(series, window1=5, window2=25):
        ma1 = DataSource.MA(series, window1).rename("MA1")
        ma2 = DataSource.MA(series, window2).rename("MA2")
        diff = ma1 - ma2
        cross = pd.Series(
                        np.where((diff>0) & (diff<0).shift().fillna(False), 1,
                            np.where((diff<0) & (diff>0).shift().fillna(False), -1, 0
                                )
                        ),
                        index = series.index, name="MA_Cross"
                )
        return pd.concat([ma1, ma2, cross], axis=1)
    
    def computeFeatureSequence(self, df):
        # feature dimension
        
        pd.options.mode.chained_assignment = None  # default='warn'

        GRM_TIMEPERIOD = self.minSequenceLength

        df['MA'] = df.groupby("SecuritiesCode").apply(lambda d:d["Close"].rolling(GRM_TIMEPERIOD, min_periods=1).mean()).reset_index("SecuritiesCode",drop=True)

        #df[["MA1", "MA2", "MA_Cross"]] = df.groupby("SecuritiesCode").apply(lambda d: DataSource.createMAFeature(d["Close"]))# .join(df["Target"].shift(-1)).groupby("MA_Cross").describe()
        df['MA1'] = df.groupby("SecuritiesCode").apply(lambda d:d["Close"].rolling(5, min_periods=1).mean()).reset_index("SecuritiesCode",drop=True)
        df['MA2'] = df.groupby("SecuritiesCode").apply(lambda d:d["Close"].rolling(25, min_periods=1).mean()).reset_index("SecuritiesCode",drop=True)
        df['MA_Cross'] = df["MA1"] - df["MA2"]
        df["Diff"] = (df["Close"] - df["Open"]) / df[["Close","Open"]].mean(axis=1)
        df["Diff_MA1"] = df["Close"] - df["MA1"]
        df["Diff_MA2"] = df["Close"] - df["MA2"]
        for i in range(1, 3):
            df["MA_Cross_lag_{:}".format(i)] = df.groupby("SecuritiesCode")["MA_Cross"].shift(i)

        df["DivMA"] = df.groupby("SecuritiesCode")["Close"].apply(DataSource.DMA)
        df["Div"] = df.groupby("SecuritiesCode")["Close"].apply(DataSource.divergence)
        df["Rsi"] = df.groupby("SecuritiesCode")["Close"].apply(DataSource.rsi)
        df = df.join(df.groupby("SecuritiesCode")["Close"].apply(DataSource.stochastic))

        df['GRM_0'] = df.groupby("SecuritiesCode").apply(lambda d:ta.MA(d["Close"], timeperiod=GRM_TIMEPERIOD, matype=0)).reset_index("SecuritiesCode",drop=True)

        df['RSI_14'] = df.groupby("SecuritiesCode").apply(lambda d:ta.RSI(d["Close"], timeperiod=14)).reset_index("SecuritiesCode",drop=True)
        df['RSI_24'] = df.groupby("SecuritiesCode").apply(lambda d:ta.RSI(d["Close"], timeperiod=24)).reset_index("SecuritiesCode",drop=True)

        df['RSI_1'] = df.groupby("SecuritiesCode").apply(lambda d:d["RSI_14"].shift(1)).reset_index("SecuritiesCode",drop=True)
        df['RSI_4'] = df.groupby("SecuritiesCode").apply(lambda d:d["RSI_14"].shift(4)).reset_index("SecuritiesCode",drop=True)
        df['RSI_7'] = df.groupby("SecuritiesCode").apply(lambda d:d["RSI_14"].shift(7)).reset_index("SecuritiesCode",drop=True)
        df['RSI_10'] = df.groupby("SecuritiesCode").apply(lambda d:d["RSI_14"].shift(10)).reset_index("SecuritiesCode",drop=True)
        df['RSI_13'] = df.groupby("SecuritiesCode").apply(lambda d:d["RSI_14"].shift(13)).reset_index("SecuritiesCode",drop=True)
        df['RSI_16'] = df.groupby("SecuritiesCode").apply(lambda d:d["RSI_14"].shift(16)).reset_index("SecuritiesCode",drop=True)

        df['ROCP'] = df.groupby("SecuritiesCode").apply(lambda d:ta.ROCP(d["Close"], timeperiod=10)).reset_index("SecuritiesCode",drop=True)
        df['ROCP15'] = df.groupby("SecuritiesCode").apply(lambda d:ta.ROCP(d["Close"], timeperiod=15)).reset_index("SecuritiesCode",drop=True)
        df['ROCP21'] = df.groupby("SecuritiesCode").apply(lambda d:ta.ROCP(d["Close"], timeperiod=21)).reset_index("SecuritiesCode",drop=True)
        df['ROCP41'] = df.groupby("SecuritiesCode").apply(lambda d:ta.ROCP(d["Close"], timeperiod=41)).reset_index("SecuritiesCode",drop=True)

        df['moment'] = df.groupby("SecuritiesCode").apply(lambda d:ta.MOM(d["Close"], timeperiod=10)).reset_index("SecuritiesCode",drop=True)
        df['CMO'] = df.groupby("SecuritiesCode").apply(lambda d:ta.CMO(d["Close"], timeperiod=14)).reset_index("SecuritiesCode",drop=True)
        df['PPO'] = df.groupby("SecuritiesCode").apply(lambda d:ta.PPO(d["Close"])).reset_index("SecuritiesCode",drop=True)
        df['SAR'] = df.groupby("SecuritiesCode").apply(lambda d:ta.SAR(d["High"], d["Low"], acceleration=0, maximum=0)).reset_index("SecuritiesCode",drop=True)

        df['DI_minus'] = df.groupby("SecuritiesCode").apply(lambda d:ta.MINUS_DI(d["High"], d["Low"], d["Close"], timeperiod=14)).reset_index("SecuritiesCode",drop=True)
        df['DI_minus1'] = df.groupby("SecuritiesCode").apply(lambda d:d["DI_minus"].shift(1)).reset_index("SecuritiesCode",drop=True)
        df['DI_minus4'] = df.groupby("SecuritiesCode").apply(lambda d:d["DI_minus"].shift(4)).reset_index("SecuritiesCode",drop=True)
        df['DI_minus7'] = df.groupby("SecuritiesCode").apply(lambda d:d["DI_minus"].shift(7)).reset_index("SecuritiesCode",drop=True)
        
        df['adx'] = df.groupby("SecuritiesCode").apply(lambda d:ta.ADX(d["High"], d["Low"], d["Close"], timeperiod=14)).reset_index("SecuritiesCode",drop=True)
        df['adx1'] = df.groupby("SecuritiesCode").apply(lambda d:d["adx"].shift(1)).reset_index("SecuritiesCode",drop=True)
        df['adx4'] = df.groupby("SecuritiesCode").apply(lambda d:d["adx"].shift(4)).reset_index("SecuritiesCode",drop=True)
        df['adx7'] = df.groupby("SecuritiesCode").apply(lambda d:d["adx"].shift(7)).reset_index("SecuritiesCode",drop=True)

        df['DI_plus'] = df.groupby("SecuritiesCode").apply(lambda d:ta.PLUS_DI(d["High"], d["Low"], d["Close"], timeperiod=14)).reset_index("SecuritiesCode",drop=True)
        df['DI_plus1'] = df.groupby("SecuritiesCode").apply(lambda d:d["DI_plus"].shift(1)).reset_index("SecuritiesCode",drop=True)
        df['DI_plus4'] = df.groupby("SecuritiesCode").apply(lambda d:d["DI_plus"].shift(4)).reset_index("SecuritiesCode",drop=True)
        df['DI_plus7'] = df.groupby("SecuritiesCode").apply(lambda d:d["DI_plus"].shift(7)).reset_index("SecuritiesCode",drop=True)

        df['apo'] = df.groupby("SecuritiesCode").apply(lambda d:ta.APO(d["Close"])).reset_index("SecuritiesCode",drop=True)
        df['apo1'] = df.groupby("SecuritiesCode").apply(lambda d:d["apo"].shift(1)).reset_index("SecuritiesCode",drop=True)
        df['apo4'] = df.groupby("SecuritiesCode").apply(lambda d:d["apo"].shift(4)).reset_index("SecuritiesCode",drop=True)
        df['apo7'] = df.groupby("SecuritiesCode").apply(lambda d:d["apo"].shift(7)).reset_index("SecuritiesCode",drop=True)

        df['natr'] = df.groupby("SecuritiesCode").apply(lambda d:ta.NATR(d["High"], d["Low"], d["Close"], timeperiod=14)).reset_index("SecuritiesCode",drop=True)

        df['variance'] = df.groupby("SecuritiesCode").apply(lambda d:ta.VAR(d["Close"], timeperiod=14)).reset_index("SecuritiesCode",drop=True)

        df['correl'] = df.groupby("SecuritiesCode").apply(lambda d:ta.CORREL(d["High"], d["Low"], timeperiod=14)).reset_index("SecuritiesCode",drop=True)
        df['TSF'] = df.groupby("SecuritiesCode").apply(lambda d:ta.TSF(d["Close"], timeperiod=14)).reset_index("SecuritiesCode",drop=True)
        df['TSF4'] = df.groupby("SecuritiesCode").apply(lambda d:d["TSF"].shift(4)).reset_index("SecuritiesCode",drop=True)
        df['TSF7'] = df.groupby("SecuritiesCode").apply(lambda d:d["TSF"].shift(7)).reset_index("SecuritiesCode",drop=True)

        return df


    ################
    def getFeature(self, start=None, end=None, iscategory=False, minSeriesLength=0 ):
        df = self.getData(start=start, end=end, minSeriesLength=minSeriesLength)
        self.printMemoryUssage('getFeature', verbose=1)
        x, y = self.computeFeature( df, iscategory=iscategory )
        return x, y
# end class DataSource
########################################
