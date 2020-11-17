import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna
from sklearn.preprocessing import MinMaxScaler
from pyts.image import GramianAngularField

__all__ = ['stationary_close', 'scale_data', 'split_data_x', 'vol_label', 'vol_label_show', 'split_data', 'stack', 'create_gasfd_defaults', 'gaxf_convert']

def stationary_close(df, col='close', diff=1):
    """
    difference in timesteps. By default this will be set to 1 which in this case is 1 hour difference. 
    
    The difference should be the time-window that is used to label data.
    """
    return np.tanh(df[col].diff(diff))

def scale_data(data, y_col=None, non_float_cols=None, scaler=None):
    """
    args:
        y_col: dependent var if y_col is in the master dataframe. 
        non_float_cols: this can be discreete vars that was will not use for GASF or GADF conversion - such as metadata. This data should be seperated
        scaler: scaler to use: example: MinMaxScaler(feature_range(n0, n-1)), RobustScaler(), etc
    """
    data = data.copy()
    cols_to_ignore = [y_col] + non_float_cols if isinstance(non_float_cols, list) else [y_col] + [non_float_cols]
    scaler = MinMaxScaler(feature_range=(0,1)) if scaler is None else scaler
    cols_to_scale = [c for c in list(data.columns) if c not in cols_to_ignore]
    for c in cols_to_scale:
        if data.iloc[:][c].dtype!=np.float64: data[c] = data.iloc[:][c].astype('float64')
        dd = data.iloc[:][c].values
        dd = dd[:, None]
        sc = scaler.fit(dd)
        data.iloc[:][c] = sc.transform(dd).squeeze(1)
    return data

def vol_label(start_idx, df, y_col, window=4, sl_pips=20, tp_pips=40):
    """
    args:
    -----------------
        stard_idx: <int> the start index should be the last idx of the time-sequence we are using as predicted features (independent vars).
                   if our time-series is from [0,1,2,3,4] then our start_idx should be 5 so our prediction label will be: [5, ... ,window]
                   therefore, the start_idx should be be the last_idx of our time-series window
        
        df:        <df> dataframe for predicting
        
        sl_pips & tp_pips: stop loss pips and take profit pips
    """
    tf = df.iloc[start_idx:start_idx+window][y_col]
    price_n = tf.iloc[0]
    upper_bound = price_n + (0.0001 * tp_pips)
    lower_bound = price_n - (0.0001 * sl_pips)
    lbls = []
    for c in tf.values:
        if c >= upper_bound: lbls.append(1)
        elif c <= lower_bound: lbls.append(2)
    return lbls[0] if len(lbls) > 0 else 0

def vol_label_show(start_idx, df, y_col, window=4, sl_pips=20, tp_pips=40):
    """
    args:
    -----------------
        stard_idx: <int> the start index should be the last idx of the time-sequence we are using as predicted features (independent vars).
                   if our time-series is from [0,1,2,3,4] then our start_idx should be 5 so our prediction label will be: [5, ... ,window]
                   therefore, the start_idx should be be the last_idx of our time-series window
        
        df:        <df> dataframe for predicting
        
        sl_pips & tp_pips: stop loss pips and take profit pips
    """
    tf = df.iloc[start_idx:start_idx+window][y_col]
    price_n = tf.iloc[0]
    upper_bound = price_n + (0.0001 * tp_pips)
    lower_bound = price_n - (0.0001 * sl_pips)
    lbls = []
    for c in tf.values:
        if c >= upper_bound: lbls.append(1)
        elif c <= lower_bound: lbls.append(2)
    plt.plot(tf.values)
    plt.plot(np.full(window, upper_bound))
    plt.plot(np.full(window, lower_bound))
    return lbls[0] if len(lbls) > 0 else 0

def split_data_x(data_dd, y_col=None, cols_to_ignore=None, window_size=24, lbl_window=4, sl_pips=20, tp_pips=40):
    """
    given data_dd (scaled data from some manner), we will return a dictionary of each feature in sequence format. By default the window will be 24 which represents 24 time steps -> in this case a single day of data. 
    
    returns: 
        ddd: <dict> representing each feature with sequences of data. [0] will contain [n0, nM] where M is the timestep size or 'window_size'. [1] will be [[0]n0+1, [0]nM+1]. therefore, each idx in the array will be a single timestep ahead of the previous one. 
    """
    c2i = [y_col] + [cols_to_ignore] if not isinstance(cols_to_ignore, list) else cols_to_ignore
    cols2ddd = [c for c in list(data_dd.columns) if c not in c2i]
    ddd = {c:[] for c in cols2ddd}
    ddd['y'] = []

    n = len(data_dd)

    start_idx=0
    end_idx=window_size
    last_idx=n-window_size

    while start_idx<last_idx:
        # grab features for time-series features
        for c in cols2ddd:
            sample = data_dd[c]
            win = sample.iloc[start_idx:end_idx].values
            ddd[c].append(win)
        
        # grab labels -> preceding window followed by window_size
        y = vol_label(end_idx, data_dd, y_col, window=lbl_window, sl_pips=sl_pips, tp_pips=tp_pips)
        ddd['y'].append(y)
        
        # increment
        start_idx+=1
        end_idx+=1
        
    return ddd

def split_data(data_dd, y_col, window_size, lbl_window, sl_pips, tp_pips):
    """
    wrappes around split_x, split_y and returns dictionary with windowed values along with y_values associated with each sequence
    """
    ddd = split_data_x(data_dd, y_col=y_col, window_size=window_size, lbl_window=lbl_window, sl_pips=sl_pips, tp_pips=tp_pips)
    return ddd


def stack(ddd):
    """
    Stacking our dictionary from our split into the appropriate shape. This is necessary for GAXF formation and for data iteration when passing through the dataset class
    
    shape:
    ----------
    [n, n_features]
        n: number of samples, in this case this should be the length of the master dataframe after dropna (if applied)
        n_features: in this case is the sequence size. Or the 'window_size' as it represents a sequence of f type features -> a feature being close, open, etc
    """
    ddd = {k:np.stack(v) for k,v in ddd.items()}
    ddd['y'] = ddd['y'][:,None]
    return ddd

def create_gasfd_defaults(size=None):
    size = 24 if size is None else size
    assert isinstance(size, int), 'size should be an int'
    gasf = GramianAngularField(image_size=size, method='summation')
    gadf = GramianAngularField(image_size=size, method='difference')
    return gasf, gadf

def gaxf_convert(ddd, c2='gasf', size=None):
    """
    GAXF, where x is either S or D. 

    We will now convert our windowed data and convert each into a GAXf format. This is what will be fed into the PyTorch model STACKED for the number of features we want to use as independent vars.
    """
    if c2.lower() not in ['gasf', 'gadf']: return
    gasf, gadf = create_gasfd_defaults(size=size)
    if c2=='gasf': temp = {k:gasf.fit_transform(v) for k,v in ddd.items() if k!='y'}
    else: temp =  {k:gadf.fit_transform(v) for k,v in ddd.items() if k!='y'}
    temp['y'] = ddd['y']
    return temp