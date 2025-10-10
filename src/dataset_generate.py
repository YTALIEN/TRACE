from benchmark.SDDObench.SDDObench import *
from benchmark.SDDObench.bench_config import *
from rbfn import construct_rbfn
from utils import *
from feature_extract import get_window_feature
from sampling import lhs
from config import Config

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

class DataStreamGenerator(Dataset):
    def __init__(
        self,
        args,
        benchmark:str='SDDObench',
        config_sddobench: dict=None,
        ):
        
        self.benchmark = benchmark
        self.dim= args.dim
        self.max_env = args.max_envs
        self.n_samples = max(args.n_samples)
        self.window_size = args.window_size
        self.feature_dim = args.fg_len
        self.args=args

        self.sddobench_config=SDDObenchConfig(config_sddobench) if config_sddobench is not None else SDDObenchConfig()
        self.original_data=self.get_original_data()

        self.seq_data=[]

    def get_original_data(self):
        match self.benchmark:
            # drf==1 --> no drift
            case "SDDObench":
                """
                    ins:[1,8]
                    drf:[1,5]
                """
                ins_num,drf_num=8,5
                origin_data=np.empty([ins_num,drf_num,self.max_env,self.n_samples,self.dim+1])
                for ins in range(1,9):
                    for drf in range(1,6):
                        params= {"num_instance": ins, "df_type": drf, "dim": self.dim, "T": self.max_env}
                        lb,ub=get_bound(ins_num)
                        for env in range(self.max_env):
                            x=lhs(self.n_samples,self.dim,lb,ub)
                            params.update(x=x, change_count=env)
                            y, params = sddobench(params,config=self.sddobench_config)
                            origin_data[ins-1,drf-1,env,:,:-1] = x
                            origin_data[ins-1,drf-1,env, :, -1] = y

            case _:
                raise ValueError(f"Benchmark {self.benchmark} is not supported.")

        return origin_data

    def get_oneins_feature(self,ins,drf):
        '''
            get the feature mat for one instance (if have many drifts, meaning one ins with one drift)
            list:len=(max_env-1)*n_windows
            move unit is the window_size
        '''
        oneins_original_data=self.original_data[ins-1,drf-1,...]
        oneins_feature=[]
        for env in tqdm(range(self.max_env-1),desc=f'F{ins}D{drf}'):
            np.random.seed(42)
            cur_env_size=np.random.choice(self.args.env_size)
            cur_data=np.random.permutation(oneins_original_data[env,...])[:cur_env_size,:]
            n_windows=int(cur_env_size/self.window_size)
            cur_model=construct_rbfn(cur_data)
            context_window=get_window_feature(window_data=np.hstack([cur_data,cur_model.predict(cur_data[:,:-1])[:,np.newaxis]]))
            window_seq=np.empty([n_windows,self.feature_dim])
            for n in range(n_windows):
                cur_window=cur_model.predict(cur_data[n * self.window_size:(n + 1) * self.window_size, :-1])
                cur_window_data=np.hstack([cur_data[n * self.window_size:(n + 1) * self.window_size], cur_window[:,np.newaxis]]) 
                cur_window_feature=get_window_feature(window_data=cur_window_data)
                window_seq[n]=cur_window_feature
            for i in range(n_windows):
                cur_data = np.delete(cur_data, slice(0,self.window_size), axis=0)
                cur_data=np.concatenate([cur_data, self.original_data[ins-1,drf-1,env + 1, i * self.window_size:(i + 1) * self.window_size ]], axis=0)
                new_window=cur_model.predict(cur_data[-self.window_size:,:-1])
                new_window_data=np.hstack([cur_data[-self.window_size:], new_window[:,np.newaxis]])
                new_window_feature=get_window_feature(window_data=new_window_data)
                window_seq=np.delete(window_seq,0,axis=0)
                window_seq=np.vstack([window_seq,new_window_feature])

                if drf==1 :
                    drift_time=0 # label==0 indicates no drift
                else:
                    drift_time=n_windows-i
                # for dynamic length
                d_len=np.random.randint(drift_time,n_windows+1)
                d_window_seq=window_seq[:d_len]
                oneins_feature.append(
                    (
                        torch.tensor(np.vstack([context_window[np.newaxis,:],d_window_seq]), dtype=torch.float32),
                        torch.tensor(drift_time, dtype=torch.float32),
                    )
                    )
        return oneins_feature
    
    def get_all_feature(self):
        '''
            get the feature mat for all instances
            list: len=ins_num*drf_num*(max_env-1)*n_windows
        '''
        match self.benchmark:
            case "SDDObench":
                for ins in range(1,9):
                    for drf in range(1,6):
                        oneins_data=self.get_oneins_feature(ins, drf)
                        self.seq_data.extend(oneins_data)
            
            case _:
                raise ValueError(f"Benchmark {self.benchmark} is not supported.")

        return self.seq_data
    
    
if __name__ == "__main__":
    args=Config(print_flag=False)
    data_stream=DataStreamGenerator(benchmark='SDDObench',args=args)
    all_data=data_stream.get_all_feature()
