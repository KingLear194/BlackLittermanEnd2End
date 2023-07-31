#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:30:10 2021

@author: jed169

A script for a one-shot training

"""

import settings
import sys
import numpy as np
import json, pickle
import time
import torch

import train_utils, trainer
import argparse

start_time = time.time()

#### parse arguments from terminal
parser = argparse.ArgumentParser(description  = 'Arguments for pytorch one-shot BLEndToEnd')

parser.add_argument('-proj','--project_name', type = str, default = 'equity', help = 'Project name')
# add job number for the case of running multiple jobs on the cluster. Note that this needs to be compatible with the file names that are saved by the respective preprocessing file.
parser.add_argument('-jobnr','--jobnumber', type = int, default = 1, help = 'job number when submitting multiple jobs to the cluster')
parser.add_argument('-printc','--printc', type =bool, default = True, help = 'print on console, otherwise save on .out file')
parser.add_argument('-time_start' , '--time_start', type=str, default = "no_timelog", help = 'get start time')

'''args for training and validation'''

# hyperparameters we optimize over in the validation phase
parser.add_argument('-rs', '--random_seed', type = int, default = 19, help = 'random seed for all packages')
parser.add_argument('-b', '--batch_size', default = 32, metavar = 'bs', help = 'total batch_size of all GPUs on the current node. Note: if want to give full batch then need to set this to a number above the data-length.')
parser.add_argument('-lr', '--learning_rate', default = 0.01, type = float, metavar = 'LR', help = 'learning rate for global optimizer')
parser.add_argument('-l2reg', '--l2regparam', default = 'none', metavar = 'L2Reg', help = 'weight for L2 regularization')
parser.add_argument('-l1reg', '--l1regparam', default = 'none', metavar = 'L1Reg_mom', help = 'weight for L1 regularization')

# hyperparameters and other modeling choices we do not optimize over in the validation phase
parser.add_argument('-validate', '--validate', type = bool, nargs = "?", const=True, default=True, metavar = 'validate', help = 'bool for validate')
parser.add_argument('-epochs', '--num_epochs', default = 10, type = int, metavar = 'Nr Epochs', help = 'Number of epochs to train')
parser.add_argument('-momentum', '--momentum', default = 0.9, type = float, metavar = 'momentum', help = 'momentum')
parser.add_argument('-scaling', '--scaling', default = True, type = bool, nargs = "?", const=True, metavar = 'xscaling', help = 'bool for scaling of relevant features')

# other
parser.add_argument('-config', '--config_file_path', default = 'none', type = str, help = 'config file name to load')
parser.add_argument('-benchmark', '--benchmark_folder', default = 'etf_cov', type = str, help = 'benchmark folder to load')
parser.add_argument('-verbose', '--verbose', default = 1, type = float, metavar = 'verbose', help = 'verbose level')
parser.add_argument('-savemodel', '--save_model', default = True, type = bool, nargs = "?", const=True,metavar = 'save model', help = 'bool for save model')


def main():
    
    args = parser.parse_args()
    # load the config parameters, this will overwrite stuff from submission script
    if args.config_file_path != 'none':
        with open(args.config_file_path, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
            
    args.savepath=f'{settings.datapath}/current_train_val_test/{args.project_name}/{args.jobnumber}/'
    
    if not args.printc: 
        sys.stdout = open(f"{args.savepath}/{args.project_name}_log_{args.time_start}.out", 'a')
    
    # retrieve architecture 
    archidefs = open(f"{args.savepath}/archidefs.pkl",'rb')
    architectures = pickle.load(archidefs)
    archidefs.close()
    args.model_args = architectures[f"{args.jobnumber}"] # this will be a list with name and list of two dicts that will contain the archi_params for the two networks
    args.view_network_params = args.model_args['view_network_params'] # this is a tuple (type_of_network, dict_network_params). type of network comes from NNmodels.py file
    args.weight_calc_params = args.model_args['weight_calc_params'] # tuple with (type_of_module to calc weights, dict_network_params) 
    args.bl_updater_params = args.model_args['bl_updater_params'] # tuple with (type of updater and dict_updater)
    
    del architectures
    
    # turn l2regparam into float
    args.global_optimizer_params = {}
    args.global_optimizer_params['name'] = args.optimizer_name
    args.global_optimizer_params['learning_rate'] = args.learning_rate
    args.global_optimizer_params['momentum'] = args.momentum
    args.global_optimizer_params['l2regparam'] = args.l2regparam
    
    args.best_mispricing_loss = np.Inf
    args.best_sharpe_ratio = -np.Inf
    
    args.path_tail = f"jnr_{args.jobnumber}_bs_{args.batch_size}_lr_{args.learning_rate}_l2reg_{args.l2regparam}_l1reg_{args.l1regparam}_rs_{args.random_seed}"
    
    main_worker(args)
    
    if not args.printc: 
        sys.stdout.close()
        

def main_worker(args):
    
    # initialize global vars and rank
    start = time.time()

    # rank of the process needs to be the global rank here of the gpu across all gpu-s
    if args.device == 'cuda':
        torch.cuda.device(0)
        use_cuda = True
    else:
        use_cuda=False
    train_utils.random_seeding(args.random_seed, use_cuda=use_cuda)
    
    init_data = trainer.InitDataloader(args)
    # initialize optimizers
    
    train_instance = trainer.TrainValBase(project_name=args.project_name, 
                                          savepath = args.savepath, 
                                          view_network_params=args.view_network_params, 
                                          weight_calc_params=args.weight_calc_params, 
                                          bl_updater_params=args.bl_updater_params, 
                                          global_optimizer_params=args.global_optimizer_params, 
                                          l1regparam=args.l1regparam, 
                                          risk_av = args.risk_av, 
                                          trad_param = args.trad_param,
                                          early_stopping = args.early_stopping,
                                          unbalanced = False if args.project_name.startswith('etf') else True,
                                          test_loader=init_data.test_loader, 
                                          train_loader=init_data.train_loader, 
                                          val_loader=init_data.val_loader, 
                                          view_features=init_data.view_features, # has the right offset already
                                          bench_covs=init_data.bench_covs,
                                          ixmap=init_data.ixmap, # has the right offset already
                                          train_criterion = args.train_criterion, 
                                          val_criterion = args.val_criterion, 
                                          test_criterion = args.test_criterion, 
                                          perf_measures = args.perf_measures,
                                          benchmark_folder=args.benchmark_folder,
                                          device = args.device,
                                          verbose = args.verbose)
    
    print("==================================================================================================================================")
    print(f'\n\nLoaded for global training.\nTraining job {args.jobnumber} with params {args.path_tail}.\n')
        
    for epoch in range(1,(args.num_epochs+1)):
        
        # train for one epoch
        train_instance.train_one_epoch(epoch)  
    
        if args.validate:
            train_instance.validate_one_epoch(epoch)
            if args.early_stopping[0] < np.Inf:
                if train_instance.stop_early:
                    break
            
    if args.validate:
        
        # test set performance 
        train_instance.testset_performance()
        #prepare results for output/saving
        train_instance.gather_performance()
        
        if args.save_model: 
            model_pth = f"{args.savepath}/Model_{args.jobnumber}_ViewNet_{args.path_tail}.pt"
            torch.save(train_instance.view_network, model_pth)
            if not args.weight_calc_params[0].endswith('solver'):
                model_pth = f"{args.savepath}/Model_{args.jobnumber}_WeightNet_{args.path_tail}.pt"
                torch.save(train_instance.weight_calculator, model_pth)
                
        dataframes = {'train_criterion': train_instance.data_loss_logger, 'val_criterion': train_instance.val_criterion_logger, 
                      'performance':train_instance.perf_measures_logger,
                      'pf_returns': train_instance.return_series,
                      'weights':train_instance.weights}
        
        for name, df in dataframes.items():
            path = f"{args.savepath}/{name}_{args.path_tail}"
            dataframes[name].to_pickle(f"{path}.pkl")
            try:
                dataframes[name].to_excel(f"{path}.xlsx")
            except ValueError as error:
                print(f"Cannot save as excel file: {path}, with error", error)
        

    end = time.time()
    time_elapsed = end-start
    hours, rem = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'\nTraining/validation/testing for project {args.project_name}, jobnumber {args.jobnumber} with {args.path_tail}\n in device {args.device} complete in {hours} hours, {minutes} minutes and {round(seconds,1)} seconds.\n')
    print("==================================================================================================================================")
    print("==================================================================================================================================")
            

if __name__ == '__main__':
    main()
