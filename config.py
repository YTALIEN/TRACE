import argparse
import pprint
import numpy as np

def get_config():
        parser=argparse.ArgumentParser()
        # ----------------------------------- environment setting --------------------------------------------
        parser.add_argument('--benchmark', type=str, default='SDDObench',help='benchmark:SDDObench,DBG')
        parser.add_argument('--dim', type=int, default=10,help='dimension of streaming data')

        #--------------------------------- drift network training ------------------------------------------
        parser.add_argument('--lr', type=float, default=5e-4,help='learning rate of optimizer')
        parser.add_argument('--weight_decay', type=float, default=1e-3,help='weight decay of optimizer: AdamW')
        parser.add_argument('--epochs', type=int, default=50,help='epochs of training drift network')
        parser.add_argument('--batch_size', type=int, default=32,help='batch size of training')

        #-------------------------------------- training data -------------------------------------------------
        parser.add_argument('--max_envs', type=int, default=60,help='max number of environments')
        parser.add_argument('--period', type=int, default=20,help='P: period of drift for the benchmark')
        parser.add_argument('--num_peaks', type=int, default=8,help='number of peaks for the benchmark')
        parser.add_argument('--env_size',type=int,default=600,help='number of samples in each environment')
        parser.add_argument('--n_samples', type=int, default=600,help='number of samples in each environment (training)')
        parser.add_argument('--window_size', type=int, default=30,help='window size to get fingerprint')
        parser.add_argument('--n_windows', type=int, default=20, help='number of windows in a data stream')
        parser.add_argument('--max_seq_len', type=int, default=21, help='maximum length of window sequence')
        parser.add_argument('--fg_len', type=int, default=7,help='discrimination feature length')

        # ----------------------------------------- DASE --------------------------------------------------------
        parser.add_argument('--test_zone',type=int, default=100, help='test zone size for DASE')
        parser.add_argument('--top_n_rate', type=float, default=0.2,help='the top n rate for archiving')
        parser.add_argument('--reuse_n_rate', type=float, default=0.4,help='the reuse rate for warm_start')
        parser.add_argument('--reuse_env', type=float, default=30,help='the number of past environments reuse')
        # ---------------------------------- EA ---------------------------------------------
        parser.add_argument('--pop_size', type=int, default=100,help='population size of evolutionary algorithm')
        parser.add_argument('--F', type=float, default=0.5 ,help='F in DE')
        parser.add_argument('--cr', type=float, default=0.9 ,help='cr in DE')
        # -------------------------- Optimization setting (fair for comparison) ---------------------------------------------
        parser.add_argument('--n_iterations', type=int, default=30,help='number of generations of evolutionary algorithm')
    
        config=parser.parse_args()
        return config
        # print(args)

def Config(user_config=None, print_flag=True):
    default_config=get_config()

    # default_config.n_samples=5*default_config.dim
    default_config.hidden_shape=int(np.sqrt(default_config.dim))
    default_config.top_n=int(default_config.top_n_rate*default_config.pop_size)
    default_config.reuse_n=int(default_config.reuse_n_rate*default_config.pop_size)
    
    if user_config is not None:
        # setting parameters by user
        for key, value in user_config.items():
            # default_config.update({key: value})
            setattr(default_config, key, value)
    if print_flag:
        pprint.pprint(vars(default_config)) 
    return default_config



    
