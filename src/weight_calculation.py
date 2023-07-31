#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 16:46:54 2022

@author: jed169


Utilities and networks for weight calculation

1. static network for balanced dataset (fixed number of assets for each date)
     For etfs will have same sequence of blocks for all dates, for balanced version of stocks dataset, we 
     use a size that is a power of 2, so that we can fit the whole date in a fixed number of batches!         
2. weight-solver similar to what we do for BL update.
    Here the network blocks are defined during runtime
    
TWO APPROACHES HOW TO GET WEIGHTS:
        1A. STATIC NETWORK USING SUFFICIENT STATISTICS FOR GAIN FUNCTION
        1B. STATIC NETWORK USING SUFFICIENT STATISTINCS FOR DEVIATION FROM FOC (SINCE CONVEX OPTIMIZATION PROBLEM)
        2A. WEIGHT_SOLVER USING THE GAINLOSS FUNCTION
        2B. WEIGHT_SOLVER USING A DIFFERENTIABLE DEVIATION FROM THE FOC
FOR 2: NEED TO MAKE SURE YOU MODEL WEIGHTS IN PARAMETER-DICT FORM, SUCH THAT 
YOU DO THE SAME TYPE OF SUB-BATCH PROCESSING, TOGETHER WITH THE RESPECTIVE PRE-WEIGHT UPDATE AS FOR THE STATIC NETWORK. 
"""

import torch
import torch.nn as nn
from torch import optim

from aux_utilities import create_idx_list
import performancemeasures as pm
from train_utils import create_weight_net_features, sub_cov, update_pre_weights

def normalize_weights(weights, L = 0.5, leverage_constraint_type = 'L2', epsilon = 1e-4):
    '''
    L is the upper bound of leverage (as epsilon becomes very small). It is applied on the sub-batch level
    '''
    if leverage_constraint_type=='L2':
        norm = torch.sqrt((weights**2).sum()+torch.tensor(epsilon))
    else:
        norm = weights.abs().sum()+torch.tensor(epsilon)
    
    weights = L*weights/norm
    return weights

class WeightCalcWrapper:
    
    def __init__(self,
                 name, args_dict):
        
        self.name = name
        print("name of weight calc is ", self.name)
        self.args_dict = args_dict
        if self.name == 'weights_net':
            self.weightCalc = WeightsNetwork(**self.args_dict)
        elif self.name=='weightdiff_net':
            self.weightCalc = WeightDiffNetwork(**self.args_dict)
        elif self.name=='weights_solver':
            self.weightCalc = WeightSolver(**self.args_dict)
        else:
            print("Wrong name for weight-calc-initialization")


class BlockWrapper(nn.Module):
    '''
    Parent class that wraps a NN block of linear layers and dropout. 
    Note: no batchnormalization because ultimately the fc layers do not process multiple sub-batches
        Inherits from nn.Module
    '''
    
    def __init__(self,
                 activation,
                 dropout,
                 bias=True):
    
        super(BlockWrapper, self).__init__()
        
        self.activation = getattr(nn,activation)
        self.dropout = dropout
        self.bias = bias
        
    def block(self,num_features, hidden_units, output_size):        
        od = nn.ModuleDict()
        od['input layer'] = nn.Linear(num_features,hidden_units[0])
        od[f'activation_{1}'] = self.activation()
    
        for i in range(1,len(hidden_units)):
            od[f'linear_{i}'] = nn.Linear(hidden_units[i-1],hidden_units[i],bias = self.bias)
            od[f'activation_{i+1}'] = self.activation()
            if self.dropout is not None: 
                od['dropout'] = nn.Dropout(self.dropout)
        # add final dense, linear layer aggregating the outcomes so far
        od['output layer'] = nn.Linear(hidden_units[-1],output_size, bias = self.bias)       
    
        return od
    
    def clear(self):
        # dummy function to streamline operations in the trainer.py
        pass
    def set_n(self,n):
        pass


class WeightsNetwork(BlockWrapper):
    
    '''
    This is the static network. It is just a FFN type of network, that processes the subatches sequentially 
    when we know at the beginning (and not only during runtime) the structure of blocks.
    Note: The inheritance diagram is nn.Module -->> BlockWrapper -->> WeightsNetwork, because 
    BlockWrapper already inherits from nn.Module, so no need to add it here. 
    '''
    
    def __init__(self,
                 n, # tuple with the blocks
                 hidden_units, # list of tuples, where each tuple is hidden_units for each block-processing
                 activation = 'SiLU', 
                 bias = True,
                 dropout = None,
                 risk_av = 0.2,
                 trad_param = 0.001,
                 leverage_constraint_type = 'L2',
                 leverage_bound = 0.1,
                 device = 'cpu'):
        super(WeightsNetwork,self).__init__(activation, dropout, bias)
        
        self.n = n
        self.dropout = dropout
        self.bias = True
        
        self.leverage_constraint_type = leverage_constraint_type
        self.leverage_bound = leverage_bound
        
        self.hidden_units = hidden_units
        
        self.risk_av = risk_av
        self.trad_param = trad_param
        self.device = device
        self.cov_pos_idx = create_idx_list(self.n)
        
        self.block_networks = nn.ModuleDict()
        for i in range(len(self.n)):
            self.block_networks[f'block_{i+1}'] = self.block(self.n[i]*(2+self.n[i]),self.hidden_units[i], self.n[i]) # (2+n)n features will be mapped to n weights
        
    def forward(self, returns, block_cov, pre_weights, asset_ids):
        '''
        Please look carefully at the update of the pre-weights in here! That the steps are correct!
        
        Note: the pre-weights should also come in batch-form at this stage
        Note: no batchnormalization applicable here because of how the input is processed (batch is always size 1 for each sub-batch)
        
        '''
        # adjust w.r.t. risk aversion and trading costs, these are the sufficient statistics (together with pre-weights) for the calculation below
        block_cov = (self.risk_av/(2*self.trad_param))*block_cov
        returns = (1/self.trad_param)*returns
        
        weights = []
        update_pre_w = False
        split_rets = torch.split(returns, self.n)
        split_pre_weights = torch.split(pre_weights.detach().clone(), self.n)
        asset_ids = torch.tensor(asset_ids).to(torch.long)
        if len(self.n)>1 and not torch.unique(asset_ids).shape[0] == pre_weights.shape[0]: # in case there are just as many different assets as pre-weights, no asset is repeated in the block
            update_pre_w = True
        asset_ids = torch.split(asset_ids, self.n)
        for i in range(len(self.n)):
            fun = self.block_networks[f"block_{i+1}"]
            # we first concatenate w.r.t. dim = 1, and then sort the values row-wise, so that we input a sequence of (ret, cov_row) to the linear layers
            x = torch.cat([split_rets[i].reshape(-1,1), 
                           sub_cov(block_cov, self.cov_pos_idx[i]), 
                           split_pre_weights[i].reshape(-1,1)], 
                          dim = 1).reshape(1,-1)
    
            for step, layer in fun.items():
                x = layer(x)
            x = x.flatten()
            x = normalize_weights(x, leverage_constraint_type=self.leverage_constraint_type, L=self.leverage_bound)
            # update pre_weights with copy of weights
            if update_pre_w and i<len(self.n)-1:
                # update split_pre_weights[i+1] for the assets that also appear in split_pre_weights[i]. 
                # recall here that asset_ids are sorted within a date 
                update_pre_weights(x.detach().clone(), split_pre_weights[i+1], asset_ids[i], asset_ids[i+1])
            
            weights.append(x)
        
        return torch.cat(weights), torch.cat(split_pre_weights)
    
class WeightDiffNetwork(BlockWrapper):
    '''
    Same idea as for WeightNetwork, but here we learn weight-pre_weight which has less features to learn from n(1+n) instead of n(2+n)
    This is based on the FOC condition from overleaf file. 
    See also note on inheritance in WeightNetwork.
    
    We could merge this easily with WeightNetwork, but we do not do it, because it may turn out this is the only one we need.
    
    For later at the loop stage: need to process the weights and pre_weights correctly at the end of the batch processing to get the pre_weights for the next batch
    '''
    
    def __init__(self,
                 n, # tuple with the blocksizes
                 hidden_units, # list of tuples, where each tuple is hidden_units for each block-processing
                 activation = 'SiLU', 
                 bias = True,
                 dropout = None,
                 leverage_constraint_type = 'L2',
                 leverage_bound = 0.1,
                 risk_av = 0.2,
                 trad_param = 0.001,
                 device = 'cpu'):
        
        '''
        Note: no batchnormalization applicable here because of how the input is processed (batch is always size 1 for each sub-batch)
        '''
        
        super(WeightDiffNetwork,self).__init__(activation, dropout, bias)
        
        self.n = n
        self.dropout = dropout
        self.bias = True
        self.activation = getattr(nn,activation)
        self.hidden_units = hidden_units
        self.device = device
        self.risk_av = risk_av
        self.trad_param = trad_param
        
        self.leverage_constraint_type = leverage_constraint_type
        self.leverage_bound = leverage_bound
        
        self.cov_pos_idx = create_idx_list(self.n)
        self.block_networks = nn.ModuleDict()
        for i in range(len(self.n)):
            '''
            Based on FOC for weight_diffs, the features are of dimension (1+n)*n, see create_weight_net_features function 
            in train_utils.py
            '''
            self.block_networks[f'block_{i+1}'] = self.block(self.n[i]*(1+self.n[i]),self.hidden_units[i], self.n[i]) # (2+n)n features will be mapped to n weights
                
        
    def forward(self, returns, block_cov, pre_weights, asset_ids):
        
        weights = []
        update_pre_w = False
        split_rets = torch.split(returns, self.n)
        split_pre_weights = torch.split(pre_weights.detach().clone(), self.n)
        asset_ids = torch.tensor(asset_ids).to(torch.long)
        if len(self.n)>1 and not torch.unique(asset_ids).shape[0] == pre_weights.shape[0]: 
            # in case there are just as many different assets as pre-weights, or just one sub-batch, no asset is repeated in the block
            update_pre_w = True
        asset_ids = torch.split(asset_ids, self.n)
        
        for i in range(len(self.n)):
            fun = self.block_networks[f"block_{i+1}"]
            adj_mean, adj_cov = create_weight_net_features(mu = split_rets[i], 
                                                           cov = sub_cov(block_cov, self.cov_pos_idx[i]), 
                                                           pre_weights = split_pre_weights[i].detach().clone(), 
                                                           risk_av = self.risk_av, trad_param = self.trad_param)
            # we first concatenate w.r.t. dim = 1, and then sort the values row-wise, so that we input a sequence of (ret, cov_row) to the linear layers
            x = torch.cat([adj_mean.reshape(-1,1), adj_cov],dim = 1).reshape(1,-1)#.flatten() # one batch here
            for step, layer in fun.items():
                x = layer(x)
            x = x.flatten()
            x = x+split_pre_weights[i].detach().clone() # add back pre_weights of the batch
            
            x = normalize_weights(x, leverage_constraint_type=self.leverage_constraint_type, L=self.leverage_bound)
            weights.append(x)
            if update_pre_w and i<len(self.n)-1:
                update_pre_weights(x.detach().clone(), split_pre_weights[i+1], asset_ids[i],asset_ids[i+1])
        
        return torch.cat(weights), torch.cat(split_pre_weights)#.detach()
    
    def clear(self):
        pass
    
class WeightsWrapper(nn.Module):
    '''
    parameter dict for the weights for each sub-batch for use in the weight-solver
    '''
    
    def __init__(self, n, leverage_constraint_type = 'L2', leverage_bound = 0.1,):
    
        super(WeightsWrapper, self).__init__()
        
        self.n = n
        self.leverage_constraint_type = leverage_constraint_type
        self.leverage_bound = leverage_bound
        self.weights = torch.nn.ParameterDict({f'block_{i+1}': torch.randn(self.n[i]) for i in range(len(self.n))})
        
    def forward(self, i):
        weights = self.weights[f"block_{i+1}"]
        weights = normalize_weights(weights = weights, L = self.leverage_bound, leverage_constraint_type=self.leverage_constraint_type)
        return weights
        

class WeightSolver:
    '''
    the pre-weights should also come in batch-form at this stage
    
    Please look carefully at the update of the pre-weights in here! That the steps are correct!
    
    loss_fn can be the gain function or the deviation from the FOC
    
    produce_weights returns the pre_weights of the batch
    
    For later at the loop stage: need to process the weights and pre_weights correctly at the end of the batch processing to get the pre_weights for the next batch
    '''
    
    def __init__(self, 
                 n,
                 nr_iterations,
                 optimizer_params,
                 risk_av, 
                 tr_costs,
                 loss_name = 'batch_data_loss', # another alternative is batch_foc_loss which is based on deviation from foc
                 device = 'cpu',
                 leverage_constraint_type = 'L2',
                 leverage_bound = 0.1,
                 verbose = False):
        
        self.n = n
        self.nr_iterations = nr_iterations
        self.optimizer_params = optimizer_params
        self.risk_av = risk_av
        self.tr_costs = tr_costs
        self.loss_name = loss_name
        self.device = device
        self.verbose = verbose
        
        self.leverage_constraint_type = leverage_constraint_type
        self.leverage_bound = leverage_bound
    
        self.loss_fn = getattr(pm, self.loss_name)
        
        self.init_calculation_tools()
    
    def init_calculation_tools(self):
        # for dependencies on self.n
        self.cov_pos_idx = create_idx_list(self.n)
        self.weights_model = WeightsWrapper(self.n, self.leverage_constraint_type, self.leverage_bound).to(self.device)
        self.optimizer = optim.Adam(self.weights_model.parameters(), **self.optimizer_params)
        
    def set_n(self,n):
        self.n = n
        self.init_calculation_tools()
    
    def forward(self, returns, block_cov, pre_weights, asset_ids): # batch_size-level variables
    
        if self.verbose: print("Using runtime weights-solver")
        
        asset_ids = torch.tensor(asset_ids).to(torch.long)
        
        for i in range(1,self.nr_iterations+1):
            if self.verbose: print("Weights_Solver: Iter ",i)
            
            with torch.autograd.set_detect_anomaly(True):
                self.optimizer.zero_grad()
                
                with torch.set_grad_enabled(True):
                    # make sure pre_weights don't have grad coming out
                    if len(self.n) == 1: # in case there is just one date in the batch
                        weights = self.weights_model.forward(0)
                    else: # there are multiple dates in the batch
                        weights = []
                        if torch.unique(asset_ids).shape[0] == pre_weights.shape[0]:
                        # there are just as many different assets as pre-weights, no asset is repeated in the block
                        # hence, no need to update the pre-weights along the way
                             weights = torch.cat([self.weights_model.forward(j) for j in range(len(self.n))])
                        else:
                        # we need to update the pre-weights sequentially because there are date repetitions in the sub-batches
                            split_asset_ids = torch.split(asset_ids, self.n)
                            split_pre_weights = torch.split(pre_weights.detach().clone(), self.n)
                            for j in range(len(self.n)-1):
                               curr_weights = self.weights_model.forward(j) 
                               weights.append(curr_weights)
                               update_pre_weights(weights[j].detach().clone(), split_pre_weights[j+1],split_asset_ids[j], split_asset_ids[j+1])
                            weights.append(self.weights_model.forward(len(self.n)-1))
                            pre_weights = torch.cat(split_pre_weights)
                            weights = torch.cat(weights)
                    loss = self.loss_fn(weights, pre_weights.detach(), returns.detach(), block_cov.detach(), self.n, self.risk_av, self.tr_costs)
                    
                    loss.backward()
                    self.optimizer.step()
        
        weights = torch.cat([self.weights_model.forward(j) for j in range(len(self.n))])
        
        return weights, pre_weights
    
    def to(self, device):
        '''
        dummy function to streamline the code in the train class
        '''
        pass
                    
    def get_trained_weights(self):
        return self.weights_model.weights
    
    def clear(self):
        del self.weights_model, self.optimizer, self.loss_fn
        
    def eval(self):
        '''
        dummy function to streamline the code in the train class
        '''
        pass
    def train(self):
        '''
        dummy function to streamline the code in the train class
        '''
        pass
    
    def load_state_dict(self, dictio):
        '''
        dummy function to streamline the code in the train class
        '''
        pass
    def state_dict(self):
        '''
        dummy function to streamline the code in the train class
        '''
        return None
       