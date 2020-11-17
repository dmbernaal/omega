from .basic_data import scale_data, split_data, stack, create_gasfd_defaults, gaxf_convert
import torch
import torch.nn as nn
from torch.utils.data import Dataset as Basedataset
from torch.utils.data import DataLoader
import numpy as np
from ta import add_all_ta_features
import pandas as pd

__all__ = ['feature_stack', 'Dataset', 'DataBunch']

def feature_stack(ddd_GAXF, y_col=None, cols_to_ignore=None):
    """
    Given ddd_GAXF where X is either s or d, we will form a new dataset which is 'stacked_features' where each feature in a single timestep is stacked -> similar to an image. 
    
    The shape of each timestep will then become
    
    shape: (n, features, h, w)
    
    where
        n: the number of timesteps
        features: all the features to stack, this can differ with cols_to_ignore added
        h: height
        w: width
        
    This is very similar to an image with c=features channels 
    """
    c2i = [y_col] + [cols_to_ignore] if not isinstance(cols_to_ignore, list) else cols_to_ignore
    cols2stack = [c for c in list(ddd_GAXF.keys()) if c not in c2i]
    len_timesteps = len(ddd_GAXF[cols2stack[0]])
    ddd_GAXF = {k:v[:,None,:,:] for k,v in ddd_GAXF.items() if k in cols2stack}
    stacked_features = np.concatenate([v for _,v in ddd_GAXF.items() if _ in cols2stack], 1)
    return stacked_features

class Dataset(Basedataset):
    """
    This is the core dataset class that is reponsible for iterating through our post-processed data. On init we will also take care of all pre-processing (scaling/etc)
    """
    def __init__(self, df, window, lbl_window, sl_pips=20, tp_pips=40, gaxf_type='gasf', ta_to_remove=None, im_size=None):
        """
        Params:
        -----------
            df: main dataframe class. For now this will only work with Oanda API data.
            window: window size for the timeseries
            lbl_window: window for prediction -> this preceds window
            gaxf_type: the type of image transformation
            im_size: size of the final image
        """
        super(Dataset, self).__init__()
        df = df.copy()
        _remove = ['trend_psar_up', 'trend_psar_down']
        if ta_to_remove != None:
            ta_to_remove = [ta_to_remove] if not isinstance(ta_to_remove, list) else ta_to_remove
            _remove = _remove + ta_to_remove
        
        ### ERROR CHECKING
        if gaxf_type.lower() not in ['gasf', 'gadf']: raise ValueError('Wrong type')
        gaxf_type=gaxf_type.lower()
        
        im_size = window if im_size is None else im_size
        if im_size > window: raise ValueError('im_size cannot exceed window size')
            
        ### LABELING
        ### Labeling here will be made via stationary_close. This can be turn into 
        ### a Regression or Classification task. We will follow through with Regression
        df['close_label'] = df['close']
        
        ### TA FEATURES
        ### For TA we will use all TA's and will exclude the ones we don't want
        df = add_all_ta_features(df, 'open', 'high', 'low', 'close', 'volume')
        df.drop(columns=_remove,axis=1,inplace=True)
        
        # dropna, the conversions won't work with NAN values
        # TODO: fillna instead, find an appropriate method for this
        df.dropna(inplace=True)
        
        ### DATA
        ### This will only work with Oanda data for now.
        data = df.iloc[:,2:].copy()
        
        ### SCALE -> using minmax
        data_dd = scale_data(data, y_col='close_label')
        
        ### DATA SPLIT
        ddd = split_data(data_dd, y_col='close_label', window_size=window, lbl_window=lbl_window, sl_pips=sl_pips, tp_pips=tp_pips)
        
        ### STACKING
        ddd = stack(ddd)
        
        ### GASX CONVERSION
        ddd = gaxf_convert(ddd, c2=gaxf_type, size=window)
        
        self.y = ddd['y']
        self.data = feature_stack(ddd, y_col='y')
        
    def __len__(self): return len(self.y)
    
    def __getitem__(self, idx):
        X = self.data[idx, :, :, :]
        y = self.y[idx]
        return torch.from_numpy(X).float(), torch.from_numpy(y).squeeze(0).long()
    
class DataBunch:
    """
    Wrapper class that creates sub Datasets for Train and Validation Splitting. 
    """
    def __init__(self, df, pct=0.2, window=24, lbl_window=4, sl_pips=20, tp_pips=40, gaxf_type='gasf', ta_to_remove=None, im_size=None):
        """
        Example:
        --------------
        ta_to_remove = ['others_dr', 'others_dlr', 'others_cr', 'momentum_rsi', 'momentum_tsi', 'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr', 'momentum_ao', 'momentum_kama', 'momentum_roc', 'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_psar_up_indicator', 'trend_psar_down_indicator', 'trend_aroon_down', 'trend_aroon_ind', 'trend_aroon_up']
        
        data = (DataBunch(df=df,
                  pct=0.2,
                  window=24,
                  lbl_window=4,
                  gaxf_type='gasf',
                  ta_to_remove=ta_to_remove)
       .bunch(bs=8))
        
        Params:
        --------------
            df: Main dataframe. This should be pre-data processing. Also known as the master dataframe
            pct: percentage split for valid
            window: window size
            gaxf_type: gasf or gadf
            ta_to_remove: technicals to not account for
            im_size: size of gaxf image
        """
        n = len(df)
        valid_n = int(n * pct)
        train_df = df.iloc[:-valid_n].copy()
        valid_df = df.iloc[-valid_n:].copy()
        
        self.train_ds = Dataset(train_df, window=window, lbl_window=lbl_window, sl_pips=sl_pips, tp_pips=tp_pips, ta_to_remove=ta_to_remove, im_size=im_size)
        self.valid_ds = Dataset(valid_df, window=window, lbl_window=lbl_window, sl_pips=sl_pips, tp_pips=tp_pips, ta_to_remove=ta_to_remove, im_size=im_size)
        
    def bunch(self, bs, num_workers=0, shuffle=False):
        self.train_dl = DataLoader(self.train_ds, batch_size=bs, num_workers=num_workers, shuffle=shuffle)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=bs, num_workers=num_workers, shuffle=False)
        return self