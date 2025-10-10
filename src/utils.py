import numpy as np
import copy


    

def get_error_rate(y_true, y_pred,sigma=1e-3):
    '''
        input：y_true: (n_samples,)
             y_pred: (n_samples,)
        output：error_rate: float
    '''
    y_tmp=np.where(np.abs(y_true)<sigma,1,y_true)
    error_rate=np.abs((y_true-y_pred)/y_tmp)
    return error_rate


def zscore_norm(data,mean=None,std=None):
    """
    Z-score normalization
    :param data: np.array, shape (n_samples, n_features)
    :return: np.array, normalized data
    """
    if mean is None or std is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1  # Avoid division by zero
    normalized_data = (data - mean) / std
    return normalized_data

def minmax_norm(data,lb,ub):
    """
    Min-max normalization
    :param data: np.array, shape (n_samples, n_features)
    :return: np.array, normalized data to [0,1]
    """
    normalized_data = (data - lb) / (ub - lb)
    return normalized_data

def remove_duplicate_data(data):
    _data = copy.deepcopy(data)
    sort_idx = _data[:, -1].argsort()
    sort_data = _data[sort_idx]
    _, uni_idx = np.unique(sort_data, axis=0, return_index=True)
    uni_data = sort_data[uni_idx]
    return uni_data

