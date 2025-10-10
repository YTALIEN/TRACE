from utils import get_error_rate
import numpy as np

def get_window_feature(window_data=None,error_rate=None):
    '''
        input：data_stream: (window_size,dim+2)
        output：window_feature: vect: len=feature_dim
    '''
    if error_rate is None:
        if window_data is None:
            raise ValueError("Window: data_stream and error_rate should not be None at the same time")
        else:
            y=window_data[:,-2]
            hy=window_data[:,-1]
            error_rate=get_error_rate(y,hy)
    window_feature=[]
    window_feature.append(np.mean(error_rate))
    window_feature.append(np.std(error_rate))
    window_feature.append(np.quantile(error_rate,0.25))
    window_feature.append(np.quantile(error_rate,0.5))
    window_feature.append(np.quantile(error_rate,0.75))
    window_feature.append(np.max(error_rate))
    window_feature.append(np.min(error_rate))

    return np.array(window_feature)