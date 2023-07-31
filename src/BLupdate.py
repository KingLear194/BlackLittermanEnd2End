#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 11:56:23 2022

@author: jed169

Comments: Here we do the BLupdate solvers.
There is also a wrapper for the bayesian case.

"""

import torch
import torch.nn as nn
from torch import optim

import bayesian_bl_update
import performancemeasures as pm


class BLUpdaterWrapper:
    '''
        Wrapper for both BLUpdater based on probability distance, 
        as well as on just traditional Bayesian update.
        Needs to have the parameter nr_iterations with =-1 if bayesian update is used. 
    '''
    def __init__(self,
                 name,
                 updater_args):
        print("name of BL updater is ", name)
        if name == 'bayesian_updater':
            self.updater = bayesian_bl_update.BayesBLupdate(**updater_args)
        elif name == 'solver_updater':
            self.updater = BLUpdater(**updater_args)


class BLmodel(nn.Module):
    
    def __init__(self,n, K):
        super(BLmodel, self).__init__()
        
        self.n = n # this is now a list of sizes (ints), all larger than K
        self.K = K
        self.B = torch.nn.ParameterDict({f'block_{i+1}': torch.nn.Parameter(torch.randn(size = (self.n[i],self.K))) for i in range(len(self.n))})
        self.d = torch.nn.ParameterDict({f'block_{i+1}': torch.randn(self.n[i]) for i in range(len(self.n))})
        
        self.mu = torch.nn.Parameter(torch.randn(sum(self.n)),requires_grad=True)
    
    def forward(self):
        mu = self.mu
        covs = []
        for key, b in self.B.items():
            covs.append(torch.matmul(b, b.t())+torch.diag_embed(torch.mul(self.d[key],self.d[key])))
        return mu, torch.block_diag(*covs)  

  
class BLUpdater:
    
    def __init__(self,
                 n, # tuple of sizes for block-matrix of cov
                 K, # shrinking factor for B-matrix for BLUpdater a la Meucci, should not be larger than number of views, and/or number of factors used in POET
                 nr_iterations,
                 optimizer_params = {'lr':0.001}, # dict of lr, weight_decay and other params if necessary
                 loss_name = 'JSD', #'KL', 'WSD2'
                 epsilon = torch.tensor(1e-5),
                 view_penalty = 1000.0,
                 device = 'cpu',
                 verbose = False):
        
        self.n = n
        self.K = K
        self.nr_iterations = nr_iterations
        self.optimizer_params = optimizer_params
        self.loss_name = loss_name
        self.epsilon=epsilon
        self.view_penalty = torch.tensor(view_penalty)
        self.device = device
        self.verbose = verbose
        
        self.init_calculation_tools()
        
    def init_calculation_tools(self):
        self.blmodel = BLmodel(self.n, self.K).to(self.device)
        self.init_probability_distance()
        
    def set_n(self,n):
        self.n = n
        self.init_calculation_tools()
        
    def init_probability_distance(self):
        
        probab_dist_wrapper = pm.ProbabDistWrapper(self.loss_name, self.n, self.epsilon, self.device)
        self.probability_distance = probab_dist_wrapper.probability_distance
    
    def produce_update(self, mu_b, cov_b, P, views):
        
        '''
        mu_b has shape nr_assets (sum(self.n)), not split according to dates
        cov_b is in block form with shape nr_assets x nr_assets
        P is of shape nr_assets x nr_views, not split according to dates
        views has shape nr_dates (len(self.n)) x nr_views
        '''
                
        self.optimizer = optim.Adam(self.blmodel.parameters(), **self.optimizer_params)
        
        if self.verbose: print(f"Produce BL update with solver and probability distance {self.loss_name}")
        
        
        self.blmodel.train()
        
        for i in range(1,self.nr_iterations):
            if self.verbose:
                print("Iter ",i)
            
            with torch.autograd.set_detect_anomaly(True):
                self.optimizer.zero_grad()
                
                with torch.set_grad_enabled(True):
                  
                    bl_params = list(self.blmodel.forward())
                    
                    # note: cov_b already has block matrix form! As does bl_params[1]
                    loss = self.probability_distance(mu_b, cov_b, *bl_params) 
                    + self.view_penalty*torch.linalg.norm(torch.matmul(P.T,bl_params[0])-views)
                
                    loss.backward()
                    self.optimizer.step()

    def get_BLupdate(self):
        self.blmodel.eval()
        return list(self.blmodel.forward())
    
    def clear(self):
        del self.optimizer, self.blmodel
        
            
