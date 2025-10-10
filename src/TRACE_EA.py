"""
    TRACE_EA
"""
import torch
import numpy as np
from TRACE import *
import DE
from sampling import lhs
from functools import partial   
from archive import *
from utils import *
from feature_extract import get_window_feature
from pathlib import Path

current_file_path = Path(__file__).resolve()
script_dir = current_file_path.parent



def TRACE_EA(samples:np.array,
             lb:int ,ub:int,
            env_archive,
            config,
            detector=None,
            context_vec=None,
            device=None,
             ):
    test_zone= config.test_zone
    n,d_=samples.shape
    d=d_-1 

    curenv=env_archive[-1]
    EW=curenv['EW']
    if detector is None:
        detector=TRACE().to(device)
        detector.load_state_dict(torch.load(f'{script_dir}/check_points/TRACE_v1.pkl'))
    
    new_env_flag,context_vec=is_new_env(EW=EW,
                                        curenv=curenv,
                                        detector=detector,
                                        samples=samples,
                                        device=device,
                                        config=config,
                                        context_vec=context_vec)
    if new_env_flag:
        test_data=lhs(test_zone,d,lb,ub)
        newenv=Env(data=samples)
        curenv.save_models(config=config)
        env_archive.append(curenv)

        curenv=make_new_env(newenv,env_archive,test_data,config)
    else:
        for ti in range(n):
            curenv['EW'].append(samples[ti])
            if curenv['EW'].update_flag:
                curenv.update_models(config=config)
    
    opt_solutions,iter_best=optimal_process(curenv,lb,ub,config)
    opt_res=opt_solutions[0]

    if (opt_res!=iter_best[-1]).any():
        iter_best[-1]=opt_res

    curenv.update_sols(opt_solutions)

    env_archive.append(curenv)

    # update the metrics for EW
    metrics_ew=calculate_metrics(curenv['EW']['data'],curenv)
    curenv['EW'].update(metrics_ew)
    env_archive.update(max_arc=config.max_arc)

    return iter_best,env_archive,new_env_flag,context_vec,detector


def optimal_process(curenv,lb,ub,config):
    arc_solutions=curenv['solutions']
    pop_size=config.pop_size
    max_iter=config.n_iterations
    d=config.dim
    F=config.F
    cr=config.cr
    pop=lhs(pop_size,d,lb,ub)
    if arc_solutions is not None:
        if len(arc_solutions)>0:
            pop=np.vstack((pop,arc_solutions))

    pop,iter_best=DE.current_to_best_1(pop,lb,ub,
                                f=partial(ensemble_models,models=curenv['models'],weights=curenv['weights']),
                                max_iter=max_iter,
                                F=F,cr=cr)


    best_inds=pop[:config.top_n]
    return best_inds,iter_best


def is_new_env(EW, curenv,detector,samples, device,config,context_vec=None):
    '''
        drift detection: detect the drift in the current environment
        input: EW: data in the current environment window
                curenv: the stored archive in the current environment
               detector: the detector network
               samples: the new samples in the current time point
               device: the device of the detector network
    '''
    max_len=config.max_seq_len
    window_seq=get_seq(EW,curenv,config,context_vec) #torch.tensor:(seq_len,feature_dim)
    detector.eval()
    with torch.no_grad():
        input_seq,mask=padding_seq(window_seq,max_len)
        input_seq=input_seq.to(device)
        mask=mask.to(device)
        output=detector(input_seq,mask=mask)
        y_pred=torch.argmax(output,dim=-1).item() # softmax:int
        # y_pred=y_pred.cpu().detach().numpy()
        if y_pred!=0:
            new_env_flag=True
        else:
            new_env_flag=False
        context_vec=get_context_vec(y_pred,EW,curenv,config)
    return new_env_flag,context_vec
        
def get_context_vec(drift_idx,EW,curenv,config):
    '''
        input: drift_idx: the index of the drift point in the window sequence
                EW: environment window
                curenv: the archive information of the current environment
        output: context_vec: (feature_dim,)
    '''
    max_len=config.max_seq_len
    window_size=config.window_size
    all_data=EW['data']
    context_window=[]
    for i in range(max_len-drift_idx):
        start=-1-i*window_size
        end=start-window_size
        context_window.append(all_data[start:end:-1])
    context_window=np.vstack(context_window) 
    context_window_error=calculate_metrics(context_window,curenv)
    context_vec=get_window_feature(error_rate=context_window_error)
    return context_vec


def padding_seq(seq,max_len):
    '''
        input: seq: (seq_len,feature_dim)
        output: input_seq: (max_len,feature_dim)
                mask: (max_len,)
    '''
    seq_len,feature_dim=seq.shape
    if seq_len>=max_len:
        input_seq=seq[-max_len:] # the neweset max_len windows are chosen 
        mask=torch.zeros(max_len,dtype=torch.bool) # mask is all false
    else:
        pad_len=max_len-seq_len
        padding=torch.zeros(pad_len, feature_dim) # padding to 0
        input_seq=torch.cat([seq,padding],dim=0) 
        mask=torch.zeros(max_len,dtype=torch.bool)
        mask[seq_len:]=True # padding region
    input_tensor=input_seq.unsqueeze(0) # (1,max_len,feature_dim)
    mask=mask.unsqueeze(0) # (1,max_len)
    return input_tensor,mask

def get_seq(EW,curenv,config,context_vec=None)->torch.tensor:
    cur_envdata=EW['data']
    window_size=config.window_size
    assert len(cur_envdata)% window_size == 0, "the length of the current environment data should be multiple of the window size"
    n_windows=len(cur_envdata)//window_size
    windows=[]
    for n in range(n_windows):
        cur_window=cur_envdata[n*window_size:(n+1)*window_size]
        window_error=calculate_metrics(cur_window,curenv)
        window_feature=get_window_feature(error_rate=window_error)
        windows.append(window_feature)
    if context_vec is None:
        context_vec=windows[0] # for the initial
    window_seq=torch.tensor(context_vec+windows,dtype=torch.float32)
    return window_seq
    
def calculate_metrics(samples,curenv):
    models=curenv['models']
    weights=curenv['weights']
    samples=np.atleast_2d(samples)
    x,y=samples[:,:-1],samples[:,-1]
    y_pred=ensemble_models(x,models,weights)
    metrics=get_error_rate(y,y_pred)
    return metrics

def ensemble_models(x:np.ndarray,models,weights:list=None):
    y_pred=0
    if not isinstance(models,list):
        models_list= [models]
    else:
        models_list = models

    len_m=len(models_list)
    if weights is None:
        if len(models_list)!=1:
            raise Exception("Number of models should be only 1 without weights!")
        else:
            weights_list=[1]*len_m
    else:
        weights_list=weights

    for mi,wi in zip(models_list,weights_list):
        yi=mi.predict(x)
        yi=yi*wi
        y_pred+=yi
    
    return y_pred

def make_new_env(newenv,env_archive,test_data,config):
    samples=newenv['EW']['data']
    newenv["models"].append(construct_rbfn(samples=samples,hidden_shape=config.hidden_shape))
    newenv['weights'].append(1)

    models_c=newenv['models'][0]
    
    newenv['weights'],newenv['models'],newenv['solutions']=weights_reuse_envs(env_archive,models_c,test_data,samples,config)

    return newenv

def weights_reuse_envs(env_archive,models_c,test_data,samples,config):
    
    '''
        weighted reuse the old envs in the archive
    '''

    max_reuse_env=config.reuse_env
    pop_size=config.pop_size
    top_n=config.top_n

    models=[]
    weights=[]
    solutions=[]

    test_k=ensemble_models(test_data,models_c)
    samples_k_x=samples[:,:-1]
    samples_k_y=samples[:,-1]

    models_old=[]
    len_env=len(env_archive)

    mapping_distance=np.empty(len_env) 
    approx_error=np.empty(len_env)


    for i in range(len_env):
        env_i=env_archive[i]
        model_i=env_i['models']

        test_mdi=ensemble_models(test_data,model_i)
        mapping_distance[i]=np.mean((test_k-test_mdi)**2)

        test_aei=ensemble_models(samples_k_x,model_i)
        approx_error[i]=np.mean((test_aei-samples_k_y)**2)
    
    metrics=approx_error+mapping_distance
    env_sort=np.argsort(metrics)

    w_c=np.clip((1-len_env/max_reuse_env),0.5,1)
    w_r=1-w_c
    len_reuse=min(len_env,max_reuse_env)
    models_mtrs=mapping_distance[env_sort[:len_reuse]]
    models_mtrs_fixed=np.where(models_mtrs==0,1e-6,models_mtrs)
    recip_mtrs=np.reciprocal(models_mtrs_fixed)
    w_old=w_r*recip_mtrs/np.sum(recip_mtrs)
    k=0
    for i in env_sort[:len_reuse]:
        env_i=env_archive[i]
        models_old.append(env_i['models'][0])
        sol_count_i=min(int(pop_size*w_old[k]),top_n)
        solutions.extend(env_i['solutions'][:sol_count_i])
        k+=1

    models.append(models_c)
    weights.append(w_c)
    models.extend(models_old)
    weights.extend(w_old)

    return weights,models,solutions

def initial_archives(samples,
                    lb,ub,
                    config,):
    env_archive=Archive()
    newenv = Env(data=samples)

    # model_0=construct_gpr(samples)
    model_0=construct_rbfn(samples,hidden_shape=config.hidden_shape)


    newenv['models'].append(model_0)
    newenv['weights'].append(1)

    solutions,_=optimal_process(newenv,lb,ub,config)
    newenv.update_sols(solutions)

    # update the metrics for EW
    metrics_ew = calculate_metrics(newenv['EW']['data'], newenv)
    newenv['EW'].update(metrics_ew)

    env_archive.append(newenv)

    return env_archive


