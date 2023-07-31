#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 20:45:24 2023

@author: jed169

Main trainer for BLEndToEnd

"""

import numpy as np, pandas as pd, copy, pickle

import settings
from sklearn.preprocessing import StandardScaler
import train_utils
import aux_utilities
import performancemeasures as pm
import NNmodels
import weight_calculation
import BLupdate
from TStraintest import TSTrainTestSplit

import torch

def dataprep(fixed, datapath, date_idx = 'date_id', scaling = True):
    
    '''
    fixed: share of validation in train_val data
    X_train_val, X_test should have columns: date_id, rix, ret_e, ret_e_b, view_weights
    view_features should have as first column date_id 
    
    Note that already here, the min_date_id may not be 1 and the min_rix may not be 0, because of benchmark cov construction, 
    together with any RNN preprocessing of features
    '''

    print('-'*40)
    print('-'*40)
    
    # start date generator
    splitdata = TSTrainTestSplit(fixed = fixed, date_idx=date_idx)   
    
    X_train_val = pd.read_pickle(f"{datapath}/X_train_val.pkl").set_index(['date_id','rix'])#.drop(columns = ['asset_id']) # set index because of get_train_val_data below
    if 'asset_id' in X_train_val: X_train_val.drop(columns = ['asset_id'], inplace= True)
    X_test = pd.read_pickle(f"{datapath}/X_test.pkl")
    if 'asset_id' in X_test: X_test.drop(columns = ['asset_id'], inplace= True)
    
    y_train_val = pd.DataFrame(np.zeros(shape = (X_train_val.shape[0],1)),index = X_train_val.index)
    y_test = np.zeros(shape = (X_test.shape[0],1)).astype(np.float32)
    
    print('-'*40)
    train_dates, val_dates = next(splitdata.split(X_train_val.reset_index())) 
    
    x_train, y_train, x_val, y_val = train_utils.get_train_val_data(X_train_val, y_train_val, train_dates, val_dates, date_idx=date_idx)
    max_train_date = x_train.date_id.max()
    max_val_date = x_val.date_id.max()
    
    y_train = y_train.to_numpy().astype(np.float32)
    y_val = y_val.to_numpy().astype(np.float32)
    
    view_features = pd.read_pickle(f"{datapath}/view_features.pkl").sort_values(by = 'date_id') # need to have column
    view_features_train = view_features[view_features['date_id']<=max_train_date]
    view_features_val = view_features.loc[(view_features['date_id']<=max_val_date) & (view_features['date_id']>max_train_date), :]
    view_features_test = view_features[view_features['date_id']>max_val_date]

    for df in [x_train, x_val, X_test, view_features_train, view_features_val, view_features_test]:
        df.drop(columns = ['date_id'], inplace =  True)
        assert ('date_id' not in df.columns) & ('asset_id' not in df.columns)
        
        
    x_train = x_train.to_numpy().astype(np.float32)
    x_val = x_val.to_numpy().astype(np.float32)
    x_test = X_test.to_numpy().astype(np.float32)
    
    if scaling == True:
        
        vscaler = StandardScaler()
        
        view_features_train = vscaler.fit_transform(view_features_train)
        view_features_val = vscaler.transform(view_features_val)
        view_features_test = vscaler.transform(view_features_test)
        view_features = np.concatenate([view_features_train, view_features_val, view_features_test], axis=0).astype(np.float32)
        
    else:
        vscaler = None
        view_features = np.concatenate([view_features_train.to_numpy(), view_features_val.to_numpy(), 
                            view_features_test.to_numpy()], dim=0).astype(np.float32)
    
    train_test_data = [x_train, y_train, x_val, y_val, x_test, y_test]

    return train_test_data, view_features, vscaler, max_train_date, max_val_date 


class DatasetSimple(torch.utils.data.Dataset):
    
    '''Typical dataset class for pytorch dataloader purposes. 
        Additionally, it covers the case where we only need x by creating fictitious y. This is not memory efficient.
        x, y should come in as np.arrays with with specifier .astype(np.float32)
    '''
    
    def __init__(self, x,y=None):
        
        self.length = x.shape[0] 
        self.nr_features = x.shape[1]
        self.x = torch.from_numpy(x)
        if y is not None:
            self.y = torch.from_numpy(y)
            self.y_len = self.y.shape[1]
        else:
            self.y = torch.zeros(size = (self.x.shape[0],))
            self.y_len = 1

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
        
    def __len__(self):
        return self.length
    
    def __nr_features__(self):
        return self.nr_features

class InitDataloader:

    def __init__(self,
                args):
        
        '''
            Note: 
            args contains:
                    - project_name
                    - jobnumber
                    - batch_size
                    - train_val_split
                    - ixmap_offset: we need to push down ixmap by the number of index entries we lose due to any RNN preprocessing and benchmark calculation; see overleaf
                    - date_id_offset: we need to push down view_features by the number of entries we lose due to any RNN preprocessing and benchmark calculation; see overleaf.
        '''
    
        print(f"Loading current train- and validation data for project {args.project_name}, jobnumber {args.jobnumber}.")
        
        self.batch_size = int(args.batch_size)
        print(f'Batch_size is {self.batch_size}.')
        
        datapath = f"{settings.datapath}/current_train_val_test/{args.project_name}/{args.jobnumber}"
        self.train_test_data, self.view_features, self.vscaler, self.max_train_date, self.max_val_date = None, None, None, None, None
        self.train_test_data, self.view_features, self.vscaler, self.max_train_date, self.max_val_date = dataprep(fixed = args.train_val_split,
                                                datapath = datapath,
                                                scaling=args.scaling)
        
        self.train_dataset = DatasetSimple(self.train_test_data[0], self.train_test_data[1])
        self.validation_dataset = DatasetSimple(self.train_test_data[2],self.train_test_data[3])
        self.test_dataset = DatasetSimple(self.train_test_data[4], self.train_test_data[5]) 
        
        # first column is date_id, second column is asset_id, third column is asset_dt_pos in case of etf and last column in case of equities is asset_dt_pos_ret
        self.ixmap = torch.Tensor(pd.read_pickle(f"{datapath}/ixmap.pkl").to_numpy()).to(torch.long) 
        # add zero rows on top due to offset
        self.ixmap = torch.cat([torch.zeros_like(self.ixmap[:int(args.ixmap_offset),:]), self.ixmap], dim=0)
        self.view_features = torch.Tensor(self.view_features) 
        # add zero rows on top due to offset
        self.view_features = torch.cat([torch.zeros_like(self.view_features[:int(args.date_id_offset),:]), self.view_features], dim=0)

        self.train_loader = torch.utils.data.DataLoader(dataset = self.train_dataset, batch_size=self.batch_size, 
                                                        drop_last=True, shuffle = False, pin_memory=True)#, 
        self.val_loader = torch.utils.data.DataLoader(dataset = self.validation_dataset, batch_size = self.batch_size, 
                                                      drop_last=True, shuffle = False, pin_memory=True) 
        self.test_loader = torch.utils.data.DataLoader(dataset = self.test_dataset, batch_size = self.batch_size,
                                                  drop_last=True, shuffle = False, pin_memory=True)
        
        if args.load_benchcovs:
            print("\nLoading benchmark covs to the device\n")
            with open(f"{args.load_benchcovs}","rb") as handle:
                self.bench_covs = pickle.load(handle)
        else:
            self.bench_covs=None

class TrainValBase:
    
    def __init__(self,
                 project_name,
                 savepath, # path where results are saved
                 view_network_params, # this is a tuple (type_of_network, dict_network_params). type of network comes from NNmodels.py file
                 weight_calc_params, # tuple with (type_of_module to calc weights, dict_network_params) 
                 bl_updater_params, # tuple with (type of updater and dict_updater_params)
                 global_optimizer_params, # dictionary of parameters, including name 
                 l1regparam, # l2regparam is inside global_optimizer_params
                 risk_av,
                 trad_param,
                 early_stopping = [np.Inf,0],
                 unbalanced = False,
                 test_loader=None, 
                 train_loader=None, 
                 val_loader=None, 
                 view_features=None, 
                 bench_covs = None,
                 ixmap=None, 
                 train_criterion = 'batch_data_loss', 
                 val_criterion = 'sharpe_ratio', 
                 test_criterion = 'batch_data_loss',
                 # performance measures to be calculated on final return series, first one should be the one used for validation
                 perf_measures = ('sharpe_ratio', 'sortino_ratio', 'calmar_ratio'), 
                 benchmark_folder='benchmark',
                 device = 'cpu',
                 verbose = 0):
        
        self.project_name = project_name
        self.savepath = savepath
        self.view_network_params = view_network_params
        self.weight_calc_params = weight_calc_params
        self.bl_updater_params = bl_updater_params
        self.global_optimizer_params = global_optimizer_params
        self.l1regparam = l1regparam
        self.risk_av = risk_av
        self.trad_param = trad_param
        self.early_stopping = early_stopping
        self.unbalanced = unbalanced
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.bench_covs = bench_covs
        self.device = device
        self.benchmark_folder = benchmark_folder
        
        self.ixmap = ixmap.to(self.device)
        self.view_features = view_features.to(self.device)
        
        asset_spread = self.ixmap[:,1].max()
        # needs zero on top, because asset_ids start from 1
        self.weights = torch.zeros(self.ixmap.shape[0]).to(self.device)
        self.pre_weights = torch.zeros(1+asset_spread).to(self.device) # offset of 1 because asset_ids start at 1. 
        self.train_criterion = train_criterion
        self.val_criterion = val_criterion
        self.test_criterion = test_criterion
        self.perf_measures_names = perf_measures
        
        self.verbose = verbose
        
        self.init_networks_solvers()
        self.init_loss()
        self.init_optimizer()
        
        self.return_series = {}
        self.used_dates = []
        self.earlystop_counter = None

    
    def init_optimizer(self):
        
        print(f"We load the optimizer {self.global_optimizer_params['name']}")
        
        self.params_to_update = list(self.view_network.parameters())
        if self.weight_calc_params[0]=='weightdiff_net' or self.weight_calc_params[0]=='weights_net':
            self.params_to_update+=list(self.weight_calculator.parameters())
        
        if isinstance(self.global_optimizer_params['l2regparam'], str):
            wd = 0.0
        else:
            wd = self.global_optimizer_params['l2regparam']
        
        if self.global_optimizer_params["name"] == 'SGD':
            self.global_optimizer = torch.optim.SGD(
                self.params_to_update, 
                lr = self.global_optimizer_params['learning_rate'],
                momentum = self.global_optimizer_params['momentum'],
                weight_decay = wd)
            
        elif self.global_optimizer_params["name"] == 'Adam':
            self.global_optimizer = torch.optim.Adam(
                self.params_to_update, 
                lr = self.global_optimizer_params['learning_rate'],
                weight_decay = wd)  
        
        elif self.global_optimizer_params["name"] == 'RMSprop':
            self.global_optimizer = torch.optim.RMSprop(
                self.params_to_update, 
                lr = self.global_optimizer_params['learning_rate'],
                momentum = self.global_optimizer_params['momentum'],
                weight_decay = wd)  
        else:
            raise ValueError("Optimizer {0} not found.".format(self.global_optimizer_params["name"]))
            return -1
        
    def init_networks_solvers(self):

        self.view_network = getattr(NNmodels, self.view_network_params[0])(**self.view_network_params[1]).to(self.device)
        self.weight_calculator = weight_calculation.WeightCalcWrapper(self.weight_calc_params[0],self.weight_calc_params[1]).weightCalc
        self.weight_calculator.to(self.device)
        
        self.bl_updater = BLupdate.BLUpdaterWrapper(*self.bl_updater_params).updater # no need for to(self.device) here, since not a nn.Module and it will be part of parameters
        
        self.best_model_weights = {}
        
    def init_loss(self):
        self.train_crit = getattr(pm, self.train_criterion)
        self.val_crit = getattr(pm, self.val_criterion)
        self.test_crit = getattr(pm, self.test_criterion)
        
        self.data_loss_logger = {}
        self.perf_measures = {meas: getattr(pm, meas) for meas in self.perf_measures_names}
        self.perf_measures_logger = {}
        
        self.val_criterion_logger = {}
        self.best_val_criterion = -np.Inf
    
    def calc_performance_measures(self,ret, phase = 'train'): # phase can be train, val, or test
        
        self.return_series[phase] = ret
        for name in self.perf_measures.keys():
            self.perf_measures_logger[name, phase] = self.perf_measures[name](ret)
        
    def rix_to_multidx(self,rix):
        '''
        Gets respective index of date_id, asset_id, asset_dt_pos from global rix
        Note: rix starts from zero in the dataset, hence no need to append zero to ixmap
        '''
        return self.ixmap[rix,:]
    
    def preprocess_x(self,x):
        channels = list(torch.split(x, [1,1,1, x.shape[1]-3], dim=1)) 
        for i in range(3): # don't flatten the P matrix
            channels[i] = channels[i].flatten()
        comp_idx = self.rix_to_multidx(channels[0].to(torch.long))
        date_ids = comp_idx[:,0]
        unique_dates = date_ids.unique()
        asset_ids=comp_idx[:,1].flatten()
        subatch_idx = train_utils.date_splitter(comp_idx)
        subatch_sizes = tuple(map(lambda x: x.shape[0], subatch_idx))
        
        # renormalize P weights to be zero-sum
        rP = torch.split(channels[-1], subatch_sizes, dim=0)
        P = []
        for p in rP:
            P.append(train_utils.normalize_zero_sum(p, dim=0)) # make columns of P matrix into zero-sum
        P = torch.cat(P, dim = 0)
        
        view_feats = self.view_features[unique_dates]
        
        pre_weights = self.pre_weights[asset_ids]
        
        b_covs, r_covs = [], []
        for i, date in enumerate(unique_dates):
        
            ret_pos = subatch_idx[i][:,2]
            
            if self.unbalanced:
                bench_pos = subatch_idx[i][:,3] #fetch asset_dt_pos_bench, as last column in the case of stocks
            else:
                bench_pos = ret_pos
            
            
            if self.bench_covs:
                # need to be saved as a dict with the same date as in the benchmark_folder, and as torch.Tensor
                rcov = self.bench_covs[int(date)+1].to(self.device)
                bcov = self.bench_covs[int(date)].to(self.device)
            else:
                rcov = train_utils.load_cov(date = int(date)+1, cov_path=f'{settings.datapath}/final/{self.benchmark_folder}/{self.project_name}_b').to(self.device)
                bcov = train_utils.load_cov(date = int(date), cov_path=f'{settings.datapath}/final/{self.benchmark_folder}/{self.project_name}_b').to(self.device)
            r_covs.append(train_utils.sub_cov(cov = rcov, idx = ret_pos, fill_empty = False, device=self.device))
            b_covs.append(train_utils.sub_cov(cov = bcov, idx = bench_pos, fill_empty = self.unbalanced, device = self.device))
            
        cov_b = torch.block_diag(*b_covs)
        cov_r = torch.block_diag(*r_covs)
        cov_b.requires_grad=False
        cov_r.requires_grad=False
        
        return channels[0].to(self.device), date_ids, channels[1], channels[2], pre_weights, P, view_feats, cov_b, cov_r, asset_ids, subatch_sizes
    
    def train_one_epoch(self, epoch):
        
        print("Current epoch: ", epoch)
        
        self.weight_calculator.train()
        self.view_network.train()
        
        self.data_loss_logger[('train',epoch)] = 0.0
        
        train_data_len = 0
        phase_dates, phase_rets, phase_weights = [], [], []
        
        for x, _ in self.train_loader:
            
            x = x.to(self.device)
            train_data_len+=x.shape[0]
                        
            self.global_optimizer.zero_grad()
            
            
            with torch.autograd.set_detect_anomaly(True):
                with torch.set_grad_enabled(True):
                    
                    rix, date_ids, ret_e, ret_e_b, pre_weights, P, view_feats, block_cov_b, block_cov_r, asset_ids, subatch_sizes = self.preprocess_x(x)
                    
                    #produce views
                    views = self.view_network(view_feats)
                    
                    #produce BL update
                    self.bl_updater.set_n(subatch_sizes) # for the case of runtime solver, otherwise dummy function 
                    self.bl_updater.produce_update(ret_e_b, block_cov_b, P, views)
                    
                    ret_bl, block_cov_bl=self.bl_updater.get_BLupdate()
                    
                    # find canonical representation of BL covariance -- recall that the permutation is only within each date.
                    ret_bl_perm, block_cov_bl_perm, perm = train_utils.block_canonical_repr(torch.cat((ret_bl.reshape(-1,1),block_cov_bl),dim=1), subatch_sizes)
                    # permute inputs for the weight calculation -- note: permutation is for each date!
                    rix_perm = rix[perm].to(torch.long)
                    ret_e_perm = ret_e[perm].detach()
                    pre_weights_perm = pre_weights[perm].detach()
                    asset_ids_perm = asset_ids[perm].detach()
                    block_cov_r_perm = train_utils.sub_cov(block_cov_r, perm, device = self.device) 
                    
                    # calculate weights and local updated vector of pre_weights
                    self.weight_calculator.set_n(subatch_sizes)
                    weights, cat_pre_weights = self.weight_calculator.forward(ret_bl_perm, block_cov_bl_perm, pre_weights_perm.detach(), asset_ids_perm.detach())
                    # calculate loss -- gradient should only flow through weights
                    data_loss = self.train_crit(weights, cat_pre_weights.detach(), ret_e_perm.detach(), block_cov_r_perm.detach(), subatch_sizes, self.risk_av, self.trad_param)
                    loss = data_loss + pm.l1_penalty(model_params = self.params_to_update, lam = self.l1regparam)
                    loss.backward()
                    self.global_optimizer.step()
                    # update global pre_weights using local pre_weights, note that repetitions do not cause an issue because cat_pre_weights is ordered according to time
                    train_utils.update_global_tensor(self.weights, rix_perm, weights.detach())
                    train_utils.update_global_tensor(self.pre_weights, asset_ids_perm, weights.detach())
                    phase_dates.append(date_ids.detach().cpu())
                    phase_rets.append(ret_e_perm.detach().cpu())
                    phase_weights.append(weights.detach().cpu())
            self.data_loss_logger[('train', epoch)]+=data_loss.detach().cpu().numpy()*x.size(0)    
        self.data_loss_logger[('train', epoch)]=self.data_loss_logger[('train', epoch)]/train_data_len
        rets = pm.calculate_pf_return(torch.cat(phase_dates), torch.cat(phase_rets), torch.cat(phase_weights))
        if epoch==1:
            self.used_dates.append(torch.cat(phase_dates).unique())
        self.calc_performance_measures(rets, 'train')
        self.val_criterion_logger[('train', epoch)] = self.val_crit(rets)
        
        if self.verbose > 0 and epoch%int(1/self.verbose) == 0:
            print(f"\nFinished training epoch {epoch}")
            print(f"Data loss in training set: {self.data_loss_logger['train', epoch]}")
            print(f"{self.val_criterion} in training set: {self.val_criterion_logger['train', epoch]}\n")
                
    
    def validate_one_epoch(self, epoch):
                        
        self.weight_calculator.eval()
        self.view_network.eval()
        
        self.data_loss_logger[('val',epoch)] = 0.0
        
        val_data_len = 0
        phase_dates, phase_rets, phase_weights = [], [], []
        
        for x, _ in self.val_loader:
            
            x = x.to(self.device)
            val_data_len+=x.shape[0]
            
            with torch.set_grad_enabled(False):
                rix, date_ids, ret_e, ret_e_b, pre_weights, P, view_feats, block_cov_b, block_cov_r, asset_ids, subatch_sizes = self.preprocess_x(x)
                
                #produce views
                views = self.view_network(view_feats)
                
                #produce BL update
                self.bl_updater.set_n(subatch_sizes) # for the case of runtime solver 
                self.bl_updater.produce_update(ret_e_b, block_cov_b, P, views)
                ret_bl, block_cov_bl=self.bl_updater.get_BLupdate()
            
                # find canonical representation of BL covariance -- recall that the permutation is only within each date.
                ret_bl_perm, block_cov_bl_perm, perm = train_utils.block_canonical_repr(torch.cat((ret_bl.reshape(-1,1),block_cov_bl),dim=1), subatch_sizes)
                # permute inputs for the weight calculation -- note: permutation is for each date!
                rix_perm = rix[perm].to(torch.long)
                ret_e_perm = ret_e[perm].detach()
                pre_weights_perm = pre_weights[perm].detach()
                asset_ids_perm = asset_ids[perm].detach()
                block_cov_r_perm = train_utils.sub_cov(block_cov_r, perm, device = self.device) 
                
                # calculate weights and local updated vector of pre_weights
                self.weight_calculator.set_n(subatch_sizes)
                weights, cat_pre_weights = self.weight_calculator.forward(ret_bl_perm, block_cov_bl_perm, pre_weights_perm, asset_ids_perm)
                
                # calculate loss -- gradient should only flow through weights
                data_loss = self.train_crit(weights, cat_pre_weights.detach(), ret_e_perm.detach(), block_cov_r_perm.detach(), subatch_sizes, self.risk_av, self.trad_param)
                train_utils.update_global_tensor(self.weights, rix_perm, weights.detach())
                train_utils.update_global_tensor(self.pre_weights, asset_ids_perm, weights.detach())
                phase_dates.append(date_ids.detach().cpu())
                phase_rets.append(ret_e_perm.detach().cpu())
                phase_weights.append(weights.detach().cpu())
                    
            self.data_loss_logger[('val', epoch)]+=data_loss.detach().cpu().numpy()*x.size(0)    
        self.data_loss_logger['val', epoch]=self.data_loss_logger[('val', epoch)]/val_data_len
        rets = pm.calculate_pf_return(torch.cat(phase_dates), torch.cat(phase_rets), torch.cat(phase_weights))
        if epoch==1:
            self.used_dates.append(torch.cat(phase_dates).unique())
        self.calc_performance_measures(rets, 'val')
        self.val_criterion_logger['val', epoch] = self.val_crit(rets)
        
        # early stopping routine
        self.stop_early = False
        if self.early_stopping[0] < np.Inf: 
            early_stopping_active = True
        else:
            early_stopping_active = False
            
        # don't save stupidly large train losses, even with initialization
        improvement = (self.val_criterion_logger[('val',epoch)] > self.best_val_criterion + self.early_stopping[1]) and (self.data_loss_logger['val',epoch] < 1000000) 
        
        if improvement:
            print(f'\nUpdating global best models weights based on validation of epoch {epoch}.\n')
            
            # best weights for the models so far
            self.best_model_weights['views'] = copy.deepcopy(self.view_network.state_dict())
            self.best_model_weights['weights'] = copy.deepcopy(self.weight_calculator.state_dict()) 
            self.best_val_criterion = self.val_criterion_logger[('val',epoch)]
            self.earlystop_counter = 0
            
        elif early_stopping_active:
            if self.earlystop_counter is None:
                self.earlystop_counter=0
            self.earlystop_counter += 1
            if self.earlystop_counter>= self.early_stopping[0]:
                self.stop_early = True 
                print(f"\n----Early stopping in epoch {epoch} with parameters:\npatience: {self.early_stopping[0]}\ndelta: {self.early_stopping[1]}")
        print(f"\nFinished validation epoch {epoch}")
        
        if self.verbose > 0 and epoch%int(1/self.verbose) == 0:
            print(f"Data loss in validation set set: {self.data_loss_logger['val', epoch]}")
            print(f"{self.val_criterion} in validation set in epoch {epoch}: {self.val_criterion_logger['val', epoch]}\n")
            print(f"Best {self.val_criterion} so far in validation set: {self.best_val_criterion}\n")
        
    def testset_performance(self):
        
        print("Entering testing phase.")

        if len(self.best_model_weights) != 0:    
           self.view_network.load_state_dict(self.best_model_weights['views'])
           if self.best_model_weights['views'] is not None:
               self.weight_calculator.load_state_dict(self.best_model_weights['weights'])
        self.data_loss_logger['test', 0] = 0.0
        
        self.weight_calculator.eval()
        self.view_network.eval()

        test_data_len = 0
        phase_dates, phase_rets, phase_weights = [], [], []
        for x, _ in self.test_loader:                  

            with torch.set_grad_enabled(False):
                x = x.to(self.device)
                test_data_len+=x.shape[0]

                with torch.set_grad_enabled(False):
                    rix, date_ids, ret_e, ret_e_b, pre_weights, P, view_feats, block_cov_b, block_cov_r, asset_ids, subatch_sizes = self.preprocess_x(x)
                                        
                    #produce views
                    views = self.view_network(view_feats)
                    
                    #produce BL update
                    self.bl_updater.set_n(subatch_sizes) # for the case of runtime solver 
                    self.bl_updater.produce_update(ret_e_b, block_cov_b, P, views)
                    ret_bl, block_cov_bl=self.bl_updater.get_BLupdate()
                    
                    # find canonical representation of BL covariance -- recall that the permutation is only within each date.
                    ret_bl_perm, block_cov_bl_perm, perm = train_utils.block_canonical_repr(torch.cat((ret_bl.reshape(-1,1),block_cov_bl),dim=1), subatch_sizes)
                    # permute inputs for the weight calculation -- note: permutation is within each date!
                    rix_perm = rix[perm].to(torch.long)
                    ret_e_perm = ret_e[perm].detach()
                    pre_weights_perm = pre_weights[perm].detach()
                    asset_ids_perm = asset_ids[perm].detach()
                    block_cov_r_perm = train_utils.sub_cov(block_cov_r, perm, device=self.device) 
                    
                    # calculate weights and local updated vector of pre_weights
                    self.weight_calculator.set_n(subatch_sizes)
                    weights, cat_pre_weights = self.weight_calculator.forward(ret_bl_perm, block_cov_bl_perm, pre_weights_perm, asset_ids)
                    
                    # calculate loss -- gradient should only flow through weights
                    data_loss = self.train_crit(weights, cat_pre_weights.detach(), ret_e.detach(), 
                                    block_cov_r_perm.detach(), subatch_sizes, self.risk_av, self.trad_param)
                    
                    # update global pre_weights using local pre_weights, note that repetitions do not cause an issue because cat_pre_weights is ordered according to time
                    train_utils.update_global_tensor(self.weights, rix_perm, weights.detach())
                    train_utils.update_global_tensor(self.pre_weights, asset_ids_perm, weights.detach())
                    phase_dates.append(date_ids.detach().cpu())
                    phase_rets.append(ret_e_perm.detach().cpu())
                    phase_weights.append(weights.detach().cpu())

                self.data_loss_logger['test', 0]+=data_loss.detach().cpu().numpy()*x.size(0)
        self.data_loss_logger['test', 0]=self.data_loss_logger['test',0]/test_data_len
        rets = pm.calculate_pf_return(torch.cat(phase_dates), torch.cat(phase_rets), torch.cat(phase_weights))
        self.used_dates.append(torch.cat(phase_dates).unique())
        self.calc_performance_measures(rets, 'test')
        self.val_criterion_logger['test',0] = self.val_crit(rets)
            
        if self.verbose > 0:
            print("\nFinished testing.")
            print(f"Data loss in test set set: {self.data_loss_logger['test', 0]}")
            print(f"Best val criterion in validation set: {self.best_val_criterion}\n")
            print(f"{self.val_criterion} in test set: {self.val_criterion_logger['test', 0]}\n")
    
    def gather_performance(self):
        
       self.return_series = aux_utilities.coll_dict_to_df(self.return_series)
       self.return_series.columns = ['phase', 'pred_ret_e']
       
       self.weights = pd.DataFrame(self.weights.detach().cpu().numpy(), columns = ['weights']).reset_index().rename(columns = {'index':'rix'})
       
       self.return_series['date_id'] = torch.cat(self.used_dates).detach().cpu().numpy()
       
       self.val_criterion_logger = pd.DataFrame.from_dict(self.val_criterion_logger, orient = 'index', columns = [f'{self.val_criterion}'])
       self.val_criterion_logger = self.val_criterion_logger.reset_index()
       self.val_criterion_logger['phase'] = self.val_criterion_logger['index'].transform(lambda x: x[0])
       self.val_criterion_logger['epoch'] = self.val_criterion_logger['index'].transform(lambda x: x[1])
       self.val_criterion_logger['epoch'] = self.val_criterion_logger['epoch'].astype('int')
       self.val_criterion_logger.drop(columns = ['index'], inplace = True)
       
       self.data_loss_logger = pd.DataFrame.from_dict(self.data_loss_logger, orient = 'index', columns = [f'{self.train_criterion}'])
       self.data_loss_logger = self.data_loss_logger.reset_index()
       self.data_loss_logger['phase'] = self.data_loss_logger['index'].transform(lambda x: x[0])
       self.data_loss_logger['epoch'] = self.data_loss_logger['index'].transform(lambda x: x[1])
       self.data_loss_logger['epoch'] = self.data_loss_logger['epoch'].astype('int')
       self.data_loss_logger.drop(columns = ['index'], inplace = True)
       
       self.perf_measures_logger[(self.val_criterion,'best_val')] = self.best_val_criterion
       self.perf_measures_logger = pd.DataFrame.from_dict(self.perf_measures_logger, orient = 'index', columns = ['value'])
       self.perf_measures_logger = self.perf_measures_logger.reset_index()
       self.perf_measures_logger['name'] = self.perf_measures_logger['index'].transform(lambda x: x[0])
       self.perf_measures_logger['phase'] = self.perf_measures_logger['index'].transform(lambda x: x[1])
       self.perf_measures_logger.drop(columns = ['index'], inplace = True)
       
       