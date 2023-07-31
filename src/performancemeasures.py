#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 11:57:37 2021

@author: jed-pitt

performance measures for BLEndToEnd

"""

import settings

import pandas as pd
import numpy as np
import math
import glob

from sklearn.metrics import mean_squared_error

import torch

from aux_utilities import create_idx_list, onehot_encoding, subatch_sizes_to_idx
from train_utils import weight_foc_loss, sub_cov


def rmse(y_true, y_pred):
    
    return mean_squared_error(y_true,y_pred, squared = False)

epsilon = 1e-10
def sharpe_ratio(ret, epsilon = epsilon):
    '''Takes ret tensor of timeseries of returns and returns the sharpe ratio in numpy format.
    Note: we do not use this for training, hence it will give back a numpy array, instead of a tensor as the mispricing loss'''
    
    returns = ret.flatten().detach().cpu().numpy()
    mean = np.mean(returns)
    stdev = returns.std()
    small = math.isclose(stdev, 0.0, abs_tol= epsilon)
    if len(returns)==1:
        print("Series of returns has just one date!")
        return np.sign(mean)*np.inf if mean>0 else 0.0
    elif small:
        print("Standard deviation of return series inside Sharpe ratio function is very small!\n")
        return np.sign(mean)*np.inf if mean>0 else 0.0
    return np.mean(returns)/stdev

def sortino_ratio(ret, epsilon = epsilon, mar = 0.0):
    '''Takes ret tensor of timeseries of returns and returns the sortino ratio in numpy format.
    Sortino ratio = (excess)return/sigma_neg, where sigma_neg = sqrt(sum((ret_e-mar)^2)/nr_observations).
    mar = minimum acceptable return
    Note: we do not use this for training, hence it will give back a numpy array, instead of a tensor as the mispricing loss'''
    
    returns = ret.flatten().detach().cpu().numpy()
    mean = np.mean(returns)
    sigma_neg = (returns[returns<0]-mar)**2
    
    if len(sigma_neg)==0:
        print("No negative returns!")
        return np.inf if mean> 0 else 0.0
    
    sigma_neg = np.sqrt(((sigma_neg-mar)**2).sum()/len(returns))
    small = math.isclose(sigma_neg, 0.0, abs_tol= epsilon)
    if small:
        print("Negative deviation of return series inside sortino ratio function is very small!\n")
        return np.inf if mean> 0 else 0.0
    
    return mean/sigma_neg 

def drawdown_series(ret):
    '''
    input: series of returns in pytorch tensor form
    output: maximum draw down of the VALUE of the portfolio, dd_series
    '''
    returns = ret.flatten().detach().cpu().numpy()
    values = np.cumprod(np.concatenate((np.array(1).reshape(1,-1), (1+returns).reshape(1,-1)),axis = 1).flatten())[1:] # calculate value of portfolio (per unit invested), throw out the first 1 at the end
    return 1-values/np.maximum.accumulate(values) # return drawdown series (number in (0,1), the higher, the worse the drawdown)
    
def mdd(ret):
    
    return drawdown_series(ret).max()

def calmar_ratio(ret, epsilon = epsilon):
    '''
    Calculates calmar ratio of a time series of returns = mean_excess_return/max_drawdown.
    Note: max_drawdown is positive, so higher calmar ratio is better.
    '''
    max_dd = drawdown_series(ret).max()
    mean = np.mean(ret.flatten().detach().cpu().numpy())
    small = math.isclose(max_dd, 0.0, abs_tol= epsilon)
    if small:
        print("Drawdown is too small!")
        return np.inf if mean> 0 else 0.0
    
    return mean/max_dd

def yearly_returns(returns):
    returns+=1
    chunks = torch.split(returns, 12)
    returns = torch.Tensor([torch.prod(x)-1 for x in chunks])
    if len(chunks[-1]!=12):
        returns[-1] = (1+returns[-1])**(12//len(chunks[-1]))-1
    
    return returns

def analyze_returns(returns, output_yearly = False):
    returns = torch.Tensor(returns.to_numpy().flatten())
    
    if output_yearly:
        returns = yearly_returns(returns)
        
    dic = {}
    dic['sharpe_ratio'] = sharpe_ratio(returns)
    dic['sortino_ratio'] = sortino_ratio(returns)
    dic['calmar_ratio'] = calmar_ratio(returns)
    
    return pd.DataFrame.from_dict(dic, orient = 'index', columns = ['value'])

def produce_port_statistics(df, ret_label="pred_ret_e", phase_label="phase"):
    """
    compute portfolio statistics of return df
    DELETE FOR FINAL VERSION SINCE PART OF PM!
    """
    stats_df = pd.DataFrame(index=df[phase_label].unique(), columns=['mean', 'median', 'sharpe', 'sortino', 'calmar', 'mdd'])
    for phase in stats_df.index:
        ret_tensor = torch.tensor(df.loc[df['phase']==phase, ret_label].to_numpy(), dtype=torch.float32)
        stats_df.loc[phase, 'mean'] = float(torch.mean(ret_tensor))
        stats_df.loc[phase, 'median'] = float(torch.median(ret_tensor))
        stats_df.loc[phase, 'sharpe'] = sharpe_ratio(ret_tensor)
        stats_df.loc[phase, 'sortino'] = sortino_ratio(ret_tensor)
        stats_df.loc[phase, 'calmar'] = calmar_ratio(ret_tensor)
        stats_df.loc[phase, 'mdd'] = max(drawdown_series(ret_tensor))

    return stats_df

def val_test_sharpe(project_name, jobnumber, val_crit, save = True):
    returnfiles = glob.glob(f"{settings.datapath}/current_train_val_test/{project_name}/{jobnumber}/performance_*_rs_*.pkl")
    dic = {}
    for file in returnfiles:
        df = pd.read_pickle(file)
        file = file.replace(f"{settings.datapath}/current_train_val_test/{project_name}/{jobnumber}/","").replace(".pkl","").replace("performance_","")
        dic[file] = []
        dic[file].append(df.loc[(df['phase']==val_crit)&(df['name']=='sharpe_ratio'),'value'].values[0])
        dic[file].append(df.loc[(df['phase']=='test')&(df['name']=='sharpe_ratio'),'value'].values[0])
    result = pd.DataFrame.from_dict(dic, orient = 'index', columns = [f'sharpe_{val_crit}', 'sharpe_test']).sort_values(by = f'sharpe_{val_crit}', ascending = False).reset_index().rename(columns = {'index':'params'})
    if save: result.to_pickle(f"{settings.datapath}/current_train_val_test/{project_name}/{jobnumber}/{project_name}_{jobnumber}_val_test_results.pkl")
    if save: result.to_excel(f"{settings.datapath}/current_train_val_test/{project_name}/{jobnumber}/{project_name}_{jobnumber}_val_test_results.xlsx")
    
    return result

def get_best_hyper_dict(df):
    best_hyper = df.iloc[0,0].split('_')
    best_hyper = dict(zip(best_hyper[::2],best_hyper[1::2]))
    best_hyper['b'] = best_hyper['bs']
    del best_hyper['bs']
    best_hyper['jobnr'] = best_hyper['jnr']
    del best_hyper['jnr']
    del best_hyper['rs']
    return best_hyper

def calc_ratios(rets, trainend, valend, return_label = 'return_mean', date_idx = 'date_id'):
    
    rets = rets[[date_idx, return_label]]
    rets_train = rets[rets[date_idx]<=trainend]
    rets_test = rets[rets[date_idx]>valend]
    rets_val = rets.loc[(rets[date_idx]<=valend) & (rets[date_idx]>trainend), :]
    perfdict = {}
    perfdict['train_monthly'] = analyze_returns(rets_train[return_label])
    perfdict['val_monthly'] = analyze_returns(rets_val[return_label])
    perfdict['test_monthly'] = analyze_returns(rets_test[return_label])
    perfdict['train_yearly'] = analyze_returns(rets_train[return_label], output_yearly=True)
    perfdict['val_yearly'] = analyze_returns(rets_val[return_label], output_yearly=True)
    perfdict['test_yearly'] = analyze_returns(rets_test[return_label], output_yearly=True)
    perf = pd.concat(list(perfdict.values()), axis = 1)
    perf.columns = list(perfdict.keys())
        
    return perf

def get_weights_returns(path_tail, seeds = (0, 10, 12, 19, 26, 100), ensemble_type = 'mean'):
    
    rets = glob.glob(f"pf_returns_{path_tail}_rs_*.pkl") 
    weights = glob.glob(f"weights_{path_tail}_rs_*.pkl") 
    assert len(rets)==len(weights) and len(rets)==len(seeds)
    
    rets_df = []
    weights_df=[]
    
    for seed in seeds:
        rets_df.append(pd.read_pickle(f"pf_returns_{path_tail}_rs_{seed}.pkl").rename(columns = {'pred_ret_e':f'return_{seed}'}).set_index(['date_id','phase']))
        weights_df.append(pd.read_pickle(f"weights_{path_tail}_rs_{seed}.pkl").rename(columns = {'weights':f'weights_{seed}'}).set_index('rix'))
    
    rets = pd.concat(rets_df, axis = 1)
    rets['return_mean'] = rets.mean(axis=1)
    weights = pd.concat(weights_df, axis = 1)
    weights['weights_mean'] = weights.mean(axis=1)
    
    weights.reset_index(inplace = True)
    rets.reset_index(inplace = True)
    assets_dates = pd.concat([pd.read_pickle("X_train_val.pkl")[['rix','date_id','asset_id']], \
                        pd.read_pickle("X_test.pkl")[['rix','date_id','asset_id']]], axis=0).merge(rets[['phase','date_id']], on = 'date_id', how = 'inner')
    
    weights = assets_dates.merge(weights, how = 'inner', on='rix')
    
    return rets, weights 
    

def simple_ensemble(filelist, trainend=100, valend=300, date_idx = 'date_id'):
    final = pd.read_pickle(filelist[0])
    final.rename(columns = {'pf_return':'pf_return_idx_1'},inplace=True)
    for i, file in enumerate(filelist[1:],2):
        df = pd.read_pickle(file)
        df.rename(columns = {'pf_return':f'pf_return_idx_{i}'}, inplace = True)
        final = final.merge(df, how = 'inner', on = date_idx)
    final.set_index(date_idx,inplace = True)
    final['median_ensemble'] = final.apply(np.median, axis=1)
    final['mean_ensemble'] = final.apply(np.mean, axis=1)
    final.reset_index(inplace = True)
    final = final[[date_idx, 'median_ensemble','mean_ensemble']]
    final_train = final[final[date_idx]<=trainend]
    final_test = final[final[date_idx]>valend]
    final_val = final.loc[(final[date_idx]<=valend) & (final[date_idx]>trainend), :]
    
    perfdict = {}
    for method in ['median', 'mean']:
        perfdict[f'perf_train_{method}'] = analyze_returns(final_train[f'{method}_ensemble'])
        perfdict[f'perf_val_{method}'] = analyze_returns(final_val[f'{method}_ensemble'])
        perfdict[f'perf_test_{method}'] = analyze_returns(final_test[f'{method}_ensemble'])
    
    return perfdict

def get_drawdown_series(returns_file, trainend = 100, valend = 200, ret_label = 'pred_ret_e_mean', date_idx = 'date_id'):
    
    rets = returns_file.copy()
    if trainend!=None:
        rets_train = rets.loc[rets[date_idx]<=trainend]
        rets_train = rets[rets[date_idx]<=trainend].copy()
        rets_train['phase'] = 'train'
        rets_val = rets.loc[(rets[date_idx]>trainend) & (rets[date_idx]<=valend),:].copy()
        rets_val['phase'] = 'val'
        rets_test = rets[rets[date_idx]>valend].copy()
        rets_test['phase'] = 'test'
    else:
        rets_train = rets.loc[rets['phase']=='train']
        rets_val = rets.loc[rets['phase']=='val']
        rets_test = rets.loc[rets['phase']=='test']
    
    def fun(ret):
        ret = torch.Tensor(np.array(ret))
        return drawdown_series(ret)
    
    rets_train['drawdown'] = fun(rets_train[ret_label])
    rets_val['drawdown'] = fun(rets_val[ret_label])
    rets_test['drawdown'] = fun(rets_test[ret_label])
    dd = pd.concat([rets_train, rets_val, rets_test], axis = 0)
    return dd
    
def get_cumrets(returns, date_idx = 'date_id', return_label='return_mean', plot=True):
    
    returns = returns[[date_idx, return_label]].copy()
    returns['retplusone'] = returns[return_label]+1
    returns['cumret'] = returns['retplusone'].cumprod()
    
    returns = returns[[date_idx,'cumret']].copy()
    start = pd.DataFrame([[returns[date_idx].min()-1, 1.0]], columns = [date_idx, 'cumret'])
    returns = pd.concat([start, returns], axis = 0)
    
    if plot:
        returns[[date_idx,'cumret']].plot(x = date_idx, title = 'cumulative return')
    
    return returns

def cagr(rets, trainend, valend, date_idx = 'date_id', ret_label = 'return_mean'):
    
    ret = rets.copy()
    cagr = {}
    ret['retplusone'] = ret[ret_label]+1
    cmgr_train = np.array((ret.loc[ret[date_idx]<=trainend,ret_label]+1)).flatten()
    len_train = len(cmgr_train)
    cmgr_train = np.power(np.prod(cmgr_train), 1/len_train)
    cmgr_val = np.array((ret.loc[(ret[date_idx]>trainend)&(ret[date_idx]<=valend),ret_label]+1)).flatten()
    len_val = len(cmgr_val)
    cmgr_val = np.power(np.prod(cmgr_val), 1/len_val)
    cmgr_test = np.array((ret.loc[ret[date_idx]>=valend,ret_label]+1)).flatten()
    len_test = len(cmgr_test)
    cmgr_test = np.power(np.prod(cmgr_test), 1/len_test)
    
    cagr['train'] = -1.0+np.power(cmgr_train,12)
    cagr['val'] = -1.0+np.power(cmgr_val,12)
    cagr['test'] = -1.0+np.power(cmgr_test,12)
    
    return cagr

def max_mean_leverage(weights, trainend, valend, date_idx = 'date_id', weights_label = 'weights_mean', plot = True):
    
    '''
    plot = True plots leverage overall and in the test set
    '''
    
    wei = weights.copy()
    result = {}
    result['train_max'] = wei.loc[wei[date_idx]<=trainend, [date_idx, weights_label]].groupby(date_idx)[weights_label].transform(lambda x: x.abs().sum()).max()
    result['val_max'] = wei.loc[(wei[date_idx]>trainend)&(wei[date_idx]<=valend), [date_idx, weights_label]].groupby(date_idx)[weights_label].transform(lambda x: x.abs().sum()).max()
    result['test_max'] = wei.loc[wei[date_idx]>valend, [date_idx, weights_label]].groupby(date_idx)[weights_label].transform(lambda x: x.abs().sum()).max()
    
    result['train_mean'] = wei.loc[wei[date_idx]<=trainend, [date_idx, weights_label]].groupby(date_idx)[weights_label].transform(lambda x: x.abs().sum()).mean()
    result['val_mean'] = wei.loc[(wei[date_idx]>trainend)&(wei[date_idx]<=valend), [date_idx, weights_label]].groupby(date_idx)[weights_label].transform(lambda x: x.abs().sum()).mean()
    result['test_mean'] = wei.loc[wei[date_idx]>valend, [date_idx, weights_label]].groupby(date_idx)[weights_label].transform(lambda x: x.abs().sum()).mean()
    
    if plot:
        wei = wei[[date_idx, weights_label]]
        wei['leverage'] = wei.groupby('date_id')[weights_label].transform(lambda x: x.abs().sum())
        wei = wei[[date_idx, 'leverage']].drop_duplicates()
        wei[wei[date_idx]>valend].set_index(date_idx).plot(title = 'leverage on test set')
        wei.set_index(date_idx).plot(title = 'leverage overall')

    return result


'''
Code for loss functions and performance measures for BLEndToEnd training
'''

def calculate_pf_return_previous(date_ids, returns, weights):
    
    '''
    given sequence of date_ids, returns and weights, 
    calculates portfolio returns for each date separately and returns the series of returns
    '''
    wret = torch.mul(weights, returns) 
    print('wret is', wret)
    shapes = onehot_encoding(date_ids).sum(dim=0) # since dates are repeated
    print("shapes is ", shapes)
    dates_split = torch.split(date_ids, list(shapes),dim=0)
    print("dates_split is ", dates_split)
    wret_split = torch.split(wret, list(shapes),dim=0)
    print('wret_split is', wret_split)
    pf_returns, _ = torch.zeros(size = (len(dates_split),)), torch.zeros(size = (len(dates_split),))
    for i in range(len(dates_split)):
        print("date ", i)
        print(f"wret_split[{i}], {wret_split[i]}")
        pf_returns[i] = wret_split[i].flatten().sum()
        
    return pf_returns

def calculate_pf_return(date_ids, returns, weights):
    
    '''
    given sequence of date_ids, returns and weights, 
    calculates portfolio returns for each date separately and returns the series of returns
    '''
    wret = torch.mul(weights, returns) 
    shapes = onehot_encoding(date_ids).to(torch.float)
    pf_returns = torch.matmul(wret,shapes)
    
    return pf_returns
    
def batch_data_loss(weights, prev_weights, returns, block_cov, subatch_sizes, risk_av, tr_cost):
    '''
    Calculates batch_loss, which is sum over subbatches of subatch-losses, divided over batch_size. 
    weights are the only ones with gradients tied to computational graph, rest should be inputed with detached or no_grad.
    
    Note: batch loss  =  -gain function from overleaf, so no sign reversal necessary.  
    '''
    er = torch.dot(weights,returns)
    batch_size = len(returns)
    trad_cost = tr_cost*torch.linalg.vector_norm(weights-prev_weights, ord = 1)
    
    split_weights = torch.split(weights, subatch_sizes)
    idx_list = subatch_sizes_to_idx(subatch_sizes)
    risk_losses = []
    for i in range(len(subatch_sizes)):
        risk_losses.append(torch.dot(torch.matmul(sub_cov(block_cov, idx_list[i]),split_weights[i]).flatten(),split_weights[i]))
    risk = 0.5*risk_av*torch.stack(risk_losses).sum()
    
    return -(er-risk-trad_cost)/batch_size
    

def batch_foc_loss(weights, prev_weights, returns, block_cov, subatch_sizes, risk_av, trad_param):
    '''
    Calculates batch FOC loss based on weight_foc_loss function for balanced dataset. Similar signature to batch_data_loss
    '''
    batch_size = len(returns)
    idx_list = subatch_sizes_to_idx(subatch_sizes)
    split_weights = torch.split(weights, subatch_sizes)
    split_pre_weights = torch.split(prev_weights, subatch_sizes)
    split_returns = torch.split(returns, subatch_sizes)
    
    subatch_losses=[]
    for i in range(len(subatch_sizes)):
        subatch_losses.append(weight_foc_loss(delta=split_weights[i]-split_pre_weights[i],
                                              mu = split_returns[i], 
                                              cov = sub_cov(block_cov, idx_list[i]), 
                                              pre_weights=split_pre_weights[i], 
                                              risk_av = risk_av, 
                                              trad_param=trad_param))
        
    return torch.stack(subatch_losses).sum()/batch_size
    
# auxiliary function for L1 regularization. This will be added eventually to data loss, before doing the respective backpropagations
def l1_penalty(model_params,lam):
    
    if isinstance(lam, str):
        return 0.0
    
    l1loss = torch.nn.L1Loss(reduction = 'sum')
    l1_norm = 0
    
    # loop is needed because zeros_like does not like generators
    for param in model_params:
        l1_norm += l1loss(param, target = torch.zeros_like(param))    
        
    return lam*l1_norm

'''
Probability Distance measures for training the BL update
'''

PI = torch.tensor(math.pi)
def wsd2(mu_x,cov_x,mu_y,cov_y, epsilon=torch.tensor(1e-5), device = 'cpu'):
    
    '''
    d(P_X,P_Y) = min_{X,Y marginals}E[|X-Y|^2], that for normally distributed variables becomes
    d = |mu_X-mu_Y|^2 + trace(cov_X+cov_Y -2(cov_X*cov_Y)^(1/2))
    
    Inspired by: https://gist.github.com/Flunzmas/6e359b118b0730ab403753dcc2a447df
    and the paper https://arxiv.org/pdf/2009.14075.pdf
    
    '''
    epsilon=epsilon.to(device)
    
    mu_part = torch.linalg.norm(mu_x-mu_y)
    mm = torch.matmul(cov_x, cov_y)
    eye = torch.eye(mm.shape[0]).to(device)
    eps = eye*epsilon
    eps = eps.to(device)
    mm = mm+eps
    s = torch.linalg.eigvals(mm).real
    sq_cov_prod_trace = torch.sqrt(torch.clamp(s,min=1e-5,max = 1e+5)).sum()
    
    trace_part = torch.trace(cov_x+cov_y) - 2*sq_cov_prod_trace
    
    return mu_part + trace_part


def kl(mu_x,cov_x, mu_y, cov_y, epsilon, device):
    
    epsilon=epsilon.to(device)
    
    eigvals_x = torch.linalg.eigvals(cov_x).real
    eigvals_y = torch.linalg.eigvals(cov_y).real
    for tens in [eigvals_x, eigvals_y]:
        aux = torch.maximum(tens, torch.zeros_like(tens))
        tens = aux+epsilon # add small constant to avoid infinite gradients from ln(x)
    
    first = torch.log(eigvals_y).sum()-torch.log(eigvals_x).sum()-mu_x.shape[0]
    
    _, Q = torch.linalg.eigh(cov_y) 
    Q = Q.real
    # much more stable eigvals than torch.linalg.eigh
    second = torch.dot(1/eigvals_y.flatten(),torch.diag(Q@cov_x@Q.t()).flatten())
    mu_diff = mu_x.flatten()-mu_y.flatten()
    aux = torch.matmul(cov_y,mu_diff)
    third = torch.dot(mu_diff,aux)
    
    return 0.5*(first+second+third)
    
def jsd(mu_x,cov_x,mu_y,cov_y, epsilon=torch.tensor(1e-5), device = 'cpu'):
    '''
    We calculate Jensen-Shannon divergence using the concept of differential entropy h (Cover-Thomas pg 244, Example 8.1.2)
    
    With the mixture distribution M = 0.5(X+Y) we get the formula (after cancellations of constants)
    
        JSD(X,Y) = h(M) - 0.5(h(X)+h(Y)) = 0.5*(ln(det(Cov_M))-0.25*[ln(det(Cov_X)+ln(det(Cov_Y)))]
        
    Here h(X) = 0.5ln(2*PI*det(Cov_X)). The constants cancel out in the calculation.
    
    Note: 
        1. We calculate ln(det(A)) for a PSD matrix A as the sum of ln of eigenvalues.
        2. Result does not depend on means!
    '''    
    epsilon=epsilon.to(device)
    
    lis = []    
    cov_m = 0.25*(cov_x+cov_y)
    
    for cov in [cov_m,cov_x,cov_y]:
        eigvals = torch.linalg.eigvals(cov).real
        aux = torch.where(eigvals>0,eigvals,0)
        eye = torch.eye(aux.shape[0]).to(device)
        eps = eye*epsilon
        eps = eps.to(device)
        aux = aux+eps # add small constant identity matrix, to avoid infinite gradients from ln(x)
        lis.append(torch.log(aux).sum())
        
    return 0.5*lis[0]-0.25*(lis[1]+lis[2])

class ProbabDistWrapper:
    
    def __init__(self,
                 loss_name,
                 n, # tuple of sub-batchsizes
                 epsilon = torch.tensor(1e-5),
                 device = 'cpu'):
        
        self.loss_name = loss_name
        self.n = n
        self.epsilon = epsilon
        self.device = device
        
        if self.loss_name == 'JSD':
            self.prob_dist = jsd
        elif self.loss_name == 'KL':
            self.prob_dist = kl
        elif self.loss_name == 'WSD2':
            self.prob_dist = wsd2 
            
    def probability_distance(self, mu_b, cov_b, mu_bl, cov_bl):
        '''
        Returns mean of loss across block
        '''
        if len(self.n)==1:
            return self.prob_dist(mu_b, cov_b, mu_bl, cov_bl, epsilon=self.epsilon, device=self.device)
        
        idx_list = create_idx_list(self.n)
        
        mu_bs = torch.split(mu_b, self.n)
        cov_bs = [sub_cov(cov_b, idx) for idx in idx_list]
        mu_bls = torch.split(mu_bl, self.n)
        cov_bls = [sub_cov(cov_bl, idx) for idx in idx_list]
        
        return torch.stack([self.prob_dist(mu_bs[i], cov_bs[i], mu_bls[i], cov_bls[i], epsilon=self.epsilon, device=self.device) for i in range(len(self.n))]).sum()/sum(self.n)


def aggregate(jnr=1, bs = 28, lr=0.001, l2reg=0.1, l1reg = 0.1, rs = 0):
    '''
    Auxiliary function to aggregate outputs of model.
    '''
    weights = pd.read_pickle(f"weights_jnr_{jnr}_bs_{bs}_lr_{lr}_l2reg_{l2reg}_l1reg_{l1reg}_rs_{rs}.pkl")
    enum_assets = pd.read_excel(f"enum_assets_{jnr}.xlsx").drop(columns = ['Unnamed: 0','asset_dt_pos'])
    enum_dates = pd.read_excel(f"enum_dates_{jnr}.xlsx").drop(columns = ['Unnamed: 0'])
    rets = pd.read_pickle(f"pf_returns_jnr_{jnr}_bs_{bs}_lr_{lr}_l2reg_{l2reg}_l1reg_{l1reg}_rs_{rs}.pkl")
    X = pd.concat([pd.read_pickle("X_train_val.pkl"), pd.read_pickle("X_test.pkl")], axis = 0)
    
    X = X.merge(enum_assets, how = 'inner', on = ['asset_id','date_id'])
    X = X.merge(enum_dates, how = 'inner', on = ['date_id'])
    X = X.merge(weights, how = 'inner', on = 'rix').merge(rets, how = 'inner', on = 'date_id')
    
    X['wret'] = X['weights']*X['ret_e']
    X['correct_ret_e'] = X.groupby('date_id')['wret'].transform('sum')
    
    return X
    
    