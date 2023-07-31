#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 17:35:39 2022

@author: jed169

TrainUtils for the BLEndToEnd 

"""

import os
import random
import numpy as np
import glob, pickle
import sklearn.preprocessing
import torch
from aux_utilities import onehot_encoding

'''
Other utilities for training
'''


def random_seeding(seed_value, use_cuda=False):
    
    random.seed(seed_value) # random vars
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if use_cuda:
        # gpu vars
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

def get_bench_rets(project, path, mindate=0, maxdate=176):
    ret_b = []
    dates = []
    ret_files = glob.glob(f"{path}/*mu*.npy")
    for file in ret_files:
        f = file.replace(path,"").replace("_mu.npy","").replace(f"/{project}_b_","")
        dates.append(int(f))
        
    dates=np.sort(np.array(dates, dtype='int'))
    dates = dates[(dates>=mindate)&(dates<=maxdate)]
    for date in dates:
        ret_b.append(np.load(f"{path}/{project}_b_{date}_mu.npy").flatten())
    
    return np.concatenate(ret_b), dates

'''
Utils for data_prep for training
'''

def get_train_val_data(X,y, train_dates, val_dates, date_idx = 'date', reset_index = True):
    
    '''
    given set of dates train_dates, val_dates, date identifier, and X,y with a multiindex containing
    dates (or other identifier), this function creates
    the respective train and validation datasets.
    
    Input:
        X, (dataframe) of predictors
        y, (dataframe) of outcomes
        train_dates, (pd.series or dataframe) of validation dates only (no other columns)
        val_dates (pd.series or dataframe) of validation dates only (no other columns),
        date_idx, (str) identifier for merging variable
        
    Output:
        4 dataframes X_train, y_train, X_val, y_val with multi-index reset, if original dataframes 
        have multiindices
    '''
    X_train = X.loc[X.index.isin(train_dates, level = date_idx)].sort_index()
    y_train = y.loc[y.index.isin(train_dates,level = date_idx)].sort_index()
    X_val = X.loc[X.index.isin(val_dates,level = date_idx)].sort_index()
    y_val = y.loc[y.index.isin(val_dates,level = date_idx)].sort_index()
    
    if reset_index:
        return X_train.reset_index(), y_train.reset_index(), X_val.reset_index(), y_val.reset_index()
    else:
        return X_train, y_train, X_val, y_val
    
def data_scaling(data, scaler_type='StandardScaler', scaler_params = {}):
    
    '''
    Takes data in numpy array or dataframe form and fit_transforms a scaler of choice to the data 
    Returns the data scaled and the scaler
    
    inputs:
        - data (np array or df): features are in columns
        - scaler_type (str): current options are 'StandardScaler', 'RobustScaler', 'QuantileTransformer', 'MinMaxScaler'
        default = StandardScaler
        - scaler_params (dict): dict of parameters for scaler, user-prescribed, default = {}
        Note: to prescribe parameters follow sklearn-docu
    '''
    
    scaler = getattr(sklearn.preprocessing, scaler_type)(**scaler_params)
    
    data_scaled = scaler.fit_transform(data)
    
    return data_scaled, scaler

def normalize_zero_sum(tens, dim = 0, eps = 1e-8):
    '''
    Function that takes a tensor and normalizes the columns/rows to be zero sum, and so that sum of abs-values is 1.
    Will be used for the view matrix P from the features.
    dim=0 makes cols zero sum, dim=1 makes rows zero_sum
    E.g. for rows:
    Calculates subtracts row-mean from each row, then divides with absolute sum of row elements.
    '''
    pattern = {}
    pattern[0], pattern[1] = (tens.shape[0],1),(1,tens.shape[1])
    tens = tens-tens.mean(dim=dim).reshape(-2*dim+1,2*dim-1).repeat(pattern[dim])
    tensabsum = torch.abs(tens).sum(dim=dim)
    tensabsum = torch.where(torch.abs(tensabsum-0.0)<eps,1.0,tensabsum)
    if dim==0:
        tens= torch.div(tens, tensabsum)
    else:
        tens=torch.div(tens.T,tensabsum).T
    
    return tens

def update_global_tensor(global_tensor, indices, new_values): 
    '''
    Note: global_tensor needs to be mutable for this to work.
    '''
    global_tensor[indices] = new_values
    
def get_from_global_tensor(global_tensor, indices):
    return global_tensor[indices]
    
def rix_to_idx(composed_idx, rix_idx):
    '''
    Gets respective index of date_id, asset_id, asset_dt_pos from global rix
    Global rix should have a row of zeros on top, and have the columns: [asset_id, date_id, asset_dt_pos]. The index running from zero
    is precisely the rix. col = 0 is asset_id, col = 1 is date_id, col = 2 is asset_dt_pos.
    Note: In light of the zero row that is appended and will be used, rix_idx should come in very exact as input here, 
        because no adjustment is made to rix_idx here
    '''
    return composed_idx[rix_idx,:]

def load_cov(date, cov_path): 
    
    return torch.Tensor(np.load(f"{cov_path}_{date}_cov.npy"))

def pickle_covs(cov_path, savepath, project_name = 'etf'):
    
    dic = {}
    covpaths = glob.glob(cov_path+"/*_cov.npy")
    for covname in covpaths:
        cov = torch.Tensor(np.load(covname))
        covname = int(covname.replace(cov_path, "").replace(f"{project_name}","").replace("equity","").replace("top","").replace("daily","").replace("_cov.npy","").replace("_b_","").replace("_",""))
        dic[covname] = cov
    
    with open(savepath, "wb") as handle:
        pickle.dump(dic, handle)
    
def date_splitter(composed_index):
    '''
    Composed_index has columns date_id, asset_id, asset_dt_pos. Here, asset_dt_pos is the position of the asset within the date, used to mask covs. 
    Will be split according to date into sub-tensors along the rows.
    Composed index comes from applying 
    '''
    date_mask = onehot_encoding(composed_index[:,0])
    return torch.split(composed_index,tuple(date_mask.sum(dim=0)), dim = 0)


def get_idx_first_unique(tens, dim=0):
    '''
    for a tensor tens, we look at unique elements along dimension dim.
    We return the indices (in the same dimension) where they first appear
    Try:
        - tensor([0,0,4,6])
        - tensor([[ 1,  2,  2,  3,  3,  3,  4,  4,  4],
                [43, 33, 43, 76, 33, 76, 55, 55, 55]])
    '''
    unique, idx, counts = torch.unique(tens,dim=dim, sorted = True, return_inverse=True,return_counts = True)
    _, idx_sorted = torch.sort(idx, stable=True)
    cum_sum = torch.cat([torch.tensor([0]),counts.cumsum(0)[:-1]])
    return idx_sorted[cum_sum]

def extract_view_features(date_view_features):
    '''
    takes tensor with structure (date_id, view_features) where date_id is possibly repeated
    and returns features for thinned tensor (in the row dimension) where date_ids are unique.
    '''
    date_unique_idx = get_idx_first_unique(date_view_features[:,0].to(torch.long))
    return date_view_features[date_unique_idx,1:]

def add_zero_rows(tens, total_nr_rows, idx = False):
    '''
    add zero rows at the bottom for tensors 1D and 2D,
    for processing through the weight network and elsewhere!
    '''
    if len(tens.shape)==2:
        if tens.shape[0]<total_nr_rows:
            tens = torch.cat((tens,torch.zeros(total_nr_rows-tens.shape[0],tens.shape[1])),0)
    else:
        # case of flattened tensor
        tens = torch.cat((tens, torch.zeros(total_nr_rows-tens.shape[0])))
        if idx: tens = tens.to(torch.long)
    return tens

def pad_sym_matrix(cov_mat, n_s):
    '''
    Pads symmetric matrix with zero rows and columns (at the indices where there is a -1).
    We use this and not torch.nn.functional.pad, because the latter introduces non-determinacies when in CUDA. See pytorch doc. 
    
    Note: This will extend the matrix if there are -1 
    '''
    if cov_mat.shape[0]<=n_s: 
        return cov_mat
    cov_mat = add_zero_rows(cov_mat, total_nr_rows = n_s)
    cov_mat = add_zero_rows(cov_mat.T, total_nr_rows = n_s)
    return cov_mat

def batch_data_loss(weights, prev_weights, returns, covs, risk_av,tr_cost, batch_size):
    '''
    SKIP THIS!
    Calculates batch_loss, which is sum over subbatches of data loss, divided over batch_size in the end. 
    Dataloss for subatch is based on equation (2.2)
    weights, prev_weights, returns are of size n_s x nr_dates of batch. 
    weights are the only ones with gradients tied to computational graph, rest is detached or no_grad
    '''
    er = torch.dot(weights,returns)
    trad_cost = tr_cost*torch.linalg.vector_norm(weights-prev_weights, ord = 1)
    cov = torch.block_diag(*covs)
    risk = 0.5*risk_av*torch.dot(torch.matmul(cov,weights).flatten(),weights.flatten())
    
    return (er-risk-trad_cost)/batch_size

def perm_inverse(permutation):
    '''
    Calculates inverse of a permutation of a list of consecutive indices.
    This is correct. There might be an issue with how the data are fed to the sub_cov in the case of fill_empty=True
    
    '''
    mi = permutation.min()
    inv = torch.empty_like(permutation)
    dev = inv.get_device()
    if dev==-1:
        dev = 'cpu'
    inv[permutation-mi] = torch.arange(int(inv.size()[0])).to(dev)
    return inv+mi
        
def sub_cov(cov, idx, fill_empty = False, device = 'cpu'):
    '''
    Select sub-covariance matrix from a covariance matrix.
    In case of fill_empty = True, we impute wherever the index in idx is = -1, which means missing asset
    The imputation procedure is as follows:
        - we calculate the mean of off-diagonal (offdiag_mean) and on-diagonal (diag_mean)
        - we fill in the columns and rows where idx=-1 with diag_mean on the diagonal and offdiag_mean off the diagonal 
    '''
    
    if not fill_empty:
        return cov[:,idx][idx,:]
    
    offdiag_mean = (cov.sum()-torch.diagonal(cov).sum())/(cov.shape[1]**2-cov.shape[1])
    diag_mean = torch.diagonal(cov).mean()
        
    zeros = torch.where(idx==-1)[0]
    nonzeros = torch.where(idx>-1)[0]
        
    new_cov = cov[:,idx[nonzeros]][idx[nonzeros],:]
    new_cov = torch.cat((offdiag_mean*torch.ones((len(zeros),new_cov.shape[1]), device=device),new_cov), dim = 0)
    new_cov = torch.cat((offdiag_mean*torch.ones((new_cov.shape[0],len(zeros)), device=device),new_cov), dim = 1)

    new_cov[range(len(zeros)), range(len(zeros))] += (diag_mean-offdiag_mean)
    perm = perm_inverse(torch.cat((zeros,nonzeros)))
    return new_cov[:,perm][perm,:]

    
'''
Tools for the canonical permutation of the covariance matrices for the weights networks.
'''

def cov_canonical_repr(cov):
    '''
    This is based on: https://gmarti.gitlab.io/ml/2019/09/01/correl-invariance-permutations-nn.html
    stable=True is for tie-breaking and means the order is preserved in case of ties.
    Returns permuted cov matrix and permutation for non-block-form matrix.
    '''
    sums = cov.sum(dim=0)
    permutation = torch.argsort(sums) # stable true creates errors in current release of pytorch
    return sub_cov(cov,permutation), permutation 


def block_cov_canonical_repr(block_cov, block_sizes):
    '''
    Produces for a block matrix of covariances the covariance representative 
    for each block, and the overall permutation. 
    Note: for our purposes, this is enough, because the block covariances will be processed 
    separately.
    '''
    perms = []
    covs = []
    idx = [torch.arange(n) for n in block_sizes]
    cov, perm = cov_canonical_repr(sub_cov(block_cov,idx[0])) 
    perms.append(perm)
    covs.append(cov)
    for i, ix in enumerate(idx[1:],1):
        ix = ix+len(idx[i-1])
        print(ix)
        cov, perm = cov_canonical_repr(sub_cov(block_cov,ix)) 
        perms.append(perm+len(idx[i-1]))
        covs.append(cov)
    
    return torch.block_diag(*covs), torch.cat(perms,dim=0).to(torch.long)


def canonical_repr(ret_cov):
    '''
    This is based in spirit on: https://gmarti.gitlab.io/ml/2019/09/01/correl-invariance-permutations-nn.html
    He just permutes covariances, because that's the input in his exercise. Here, the input is (returns, covariances), 
    so we permute the full features
    Returns permuted returns, cov matrix and permutation for non-block-form matrix.
    '''
    sums = ret_cov.sum(dim=1)
    permutation = torch.argsort(sums)
    rets = ret_cov[:,0].flatten()[permutation]
    cov = sub_cov(ret_cov[:,1:],permutation)
    return rets, cov, permutation

def block_canonical_repr(block_feat, block_sizes):
    '''
    Produces for a block matrix of (returns, covariances) the permuted returns and covariance representative 
    for each block, and the overall permutation. 
    Note: for our purposes, this is enough, because the returns and block covariances will be processed 
    separately for each date.
    '''
    perms = []
    rets = []
    covs = []
    idx = [torch.arange(n) for n in block_sizes]
    ret, cov, perm = canonical_repr(torch.cat((block_feat[idx[0],0].reshape(-1,1),sub_cov(block_feat[:,1:],idx[0])), dim=1)) 
    perms.append(perm)
    covs.append(cov)
    rets.append(ret)
    for i, ix in enumerate(idx[1:],1):
        temp = sum([len(idx[y]) for y in range(i)])
        ix = ix+temp 
        bl_feat = torch.cat((block_feat[ix,0].reshape(-1,1),sub_cov(block_feat[:,1:],ix)), dim=1)
        ret, cov, perm = canonical_repr(bl_feat) 
        new_perm = perm+temp
        perms.append(new_perm)
        covs.append(cov)
        rets.append(ret)
            
    return torch.cat(rets), torch.block_diag(*covs), torch.cat(perms,dim=0).to(torch.long)
    

'''
Utils for pre-weight updating within the weight calculators. 
'''
def get_idx_common(one, two, sorted_tens = False):
    '''
    for two index tensors (torch.long type) with unique and sorted or unsorted values (decided by variable sorted_tens),
    it finds the indices of the common values in each one. It returns the indices in the original vectors one and two 
    of the common values.
    '''
    if not sorted_tens:
        one, one_idx = torch.sort(one)
        two, two_idx = torch.sort(two) 
    one_len, two_len = len(one), len(two)
    cat = torch.cat((one, two))
    cat_un, inverse, counts = torch.unique(cat, return_inverse=True, return_counts=True)
    idx_cts = torch.where(counts==2,1,0).nonzero()
    if len(idx_cts)==0: # there is empty intersection, return trivial indices
        return [torch.empty(size = (0,one_len)), torch.empty(size = (0,two_len))]
    else: 
        intersection_indicator = torch.sum(torch.cat([torch.where(inverse == idx_cts[i],1,0).reshape(1,-1) for i in range(len(idx_cts))],dim=0), dim=0)
        intersection_indicator=torch.split(intersection_indicator, (one_len,two_len))
        common_idx = [tens.nonzero().flatten() for tens in intersection_indicator]
        if sorted_tens:
            return common_idx
        else:
            return one_idx[common_idx[0]], two_idx[common_idx[1]]


def update_pre_weights(new_weights_old_date, pre_weights, asset_ids_old, asset_ids_new):
    '''
    updates pre_weights within a batch. This is tied to the implementation of the WeightsNetwork in weight_calculation.py !
    pre_weights needs to be a mutable object so that it can be updated in place!
    '''
    
    idx_old, idx_new = get_idx_common(asset_ids_old, asset_ids_new)
    if len(idx_old)>0:
        pre_weights[idx_new] = new_weights_old_date[idx_old].detach().clone()

'''
Utils for weight network
'''

def create_weight_net_features(mu, cov, pre_weights, risk_av, trad_param):
    
    '''
    Features that are used in the FOC condition for the convex optimization problem of the weights-calculation.
    This is for the balanced case.
    '''
    
    mu = (1/trad_param)* mu
    cov = (risk_av/trad_param)*cov
    
    return mu-cov@pre_weights, cov # the first object is adjusted mean, the second adjusted covariance


def weight_foc_loss(delta, mu, cov, pre_weights, risk_av, trad_param):
    
    '''
    Square of deviation from FOC for weights based on delta. See overleaf file for the FOC for weights.
    This is for one subatch, and used downstream to calculate batch foc loss. 
   '''
    
    adj_mean, adj_cov  = create_weight_net_features(mu, cov, pre_weights, risk_av, trad_param)
    deviation = ((adj_cov@delta+torch.sign(delta)-adj_mean)**2).sum()
    
    return deviation

