import os
import pandas as pd
import numpy as np
from benchmark.SDDObench.SDDObench import *
from benchmark.SDDObench.bench_config import SDDObenchConfig
from sampling import lhs
from config import Config
from TRACE_EA import TRACE_EA, initial_archives
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings("ignore")

sddobench_config=SDDObenchConfig()
metric_flag=True
MAX_RUN = 11
INST=[1,2,3,4,5,6,7,8]
DRFS=[1,2,3,4,5]
user_config={
    'dim':10,
    'benchmark':'SDDObench',
}
config=Config(user_config)
    
for run in range(MAX_RUN):
    # the following setting is suit for SDDObench
    for ins in INST:
        for drf in DRFS:
            T=config.max_envs
            initial_peaks=config.num_peaks
            P=config.period
            max_iter=config.n_iterations
            np.random.seed(42)
            env_size=np.empty(T)
            for i in range(T):
                env_size[i]=np.random.choice(config.env_size)
            chunk_size=config.window_size
            d=config.dim
            
            
            lb, ub = get_bound(ins)
            X = lhs(max(config.env_size), d, lb, ub)
            prob_params = {
                'peak_info': None, 
                'delta_info': None, 
                'num_peaks': initial_peaks, 
                'T': T, 
                'P': P,

            }
            prob_params.update(num_instance=ins, df_type=drf, dim=d)
            env_archive= None
            context_vec=None
            detector=None
            delt_info = []
            env_pbar=tqdm(
                range(T),
                total=T,
                desc=f'run:{run} SDDObench-F{ins}D{drf}',
                bar_format='{l_bar}{bar:20}{r_bar}'
            )
                
            for env in env_pbar:
                # initial environment
                cur_env_size=env_size[env].astype(int)
                assert cur_env_size % chunk_size == 0, 'env_size should be divisible by chunk_size'
                N=int(cur_env_size/chunk_size) 
                np.random.seed(42+env)
                x=np.random.permutation(X)[:cur_env_size,:]
                prob_params.update(x=x,change_count=env)
                y,prob_params= sddobench(prob_params,sddobench_config)
                delt_info.append(prob_params['delta_info'][1]) 
                np.random.seed()

                # np.random.seed(42)
                samples = np.hstack((x, y[:, np.newaxis]))
                perm_idx=np.random.permutation(cur_env_size)
                perm_splits=np.array_split(perm_idx, N)
                # np.random.seed()

                # initial before the stream data
                if env==0:
                    env_archive = initial_archives(samples, lb, ub, config)
                for t in range(N):
                    samples_t=samples[perm_splits[t]]
                    iter_best,env_archive,drift_eval,context_vec,detector= TRACE_EA(
                                                                samples=samples_t,
                                                                lb=lb,ub=ub,
                                                                env_archive=env_archive,
                                                                config=config,
                                                                detector=detector,
                                                                context_vec=context_vec,
                                                                device='cuda:0' if torch.cuda.is_available() else 'cpu',
                                                                )


