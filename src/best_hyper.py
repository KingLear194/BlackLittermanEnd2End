#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:13:19 2023

@author: jed169


Get best hyper parameters given a project, and a list of jobnumbers in a project.
Note: this creates and saves as an intermediate result val_test file which ranks all trained models according to best sharpe ratio 
in validation set.
"""

import sys
import json
import settings
import pandas as pd
import argparse
import performancemeasures as pm

def df_from_list(filenames, axis = 0):
    
    dfs = [pd.read_pickle(name) for name in filenames]
    df = pd.concat(dfs, axis)
    
    return df


parser = argparse.ArgumentParser(description  = 'Arguments for fetching best hyper parameters across different ')
parser.add_argument('-proj','--project_name', type = str, default = 'etf', help = 'Project name')
parser.add_argument('-val_crit','--val_criterion', type = str, default = 'sharpe_val', help = 'Criterion to choose models from.')
parser.add_argument('-jobnr','--jobnumbers', nargs="+", type = int, default = 10, help = 'job numbers to get the train/val results from')

'''
#######################################
Read off from terminal of the node and prepare the arguments for mp.spawn and later dist.init_process_group
NOTE: other args (in particular epochs) that are not hyperparameters and architecture, need to be put through the terminal
#######################################
'''
def main():
    
    args = parser.parse_args()
    
    if args.val_criterion == 'sharpe_best_val' or args.val_criterion=='sharpe_val': #for sharpe ratio pick the highest
        ascending = False
    else:
        ascending = True
    
    savepath=f"{settings.datapath}/current_train_val_test/{args.project_name}/"
    dfs = []
    for jobnr in args.jobnumbers:
        dfs.append(pm.val_test_sharpe(args.project_name, jobnr, save = True, val_crit = args.val_criterion.replace("sharpe_","")))

    valtest = pd.concat(dfs, axis = 0).sort_values(by=f"{args.val_criterion}", ascending=ascending)
    valtest.to_pickle(f"{settings.datapath}/current_train_val_test/{args.project_name}/{args.project_name}_all_val_test_results.pkl")
    hyperparams_dict = pm.get_best_hyper_dict(valtest)
    with open(f"{savepath}/{args.project_name}_best_hyperparams.json", "w") as outfile:
        json.dump(hyperparams_dict, outfile)
        
    intparams = ['b', 'jobnr']
    
    best_hyper_string = ""
    for key, val in hyperparams_dict.items():
        best_hyper_string+=f"-{key} "
        if key in intparams:
            best_hyper_string+=f"{int(val)} "
        else:
            best_hyper_string+=f"{val} "
    
    
    '''
    Need to pass this further on to the other file that does the ensembling
    '''

    return best_hyper_string, hyperparams_dict['jobnr']

if __name__ == '__main__':
    best = str(main()[0])
    jobnr = str(main()[1])
    print(best,',',jobnr)
    sys.exit(0)
    

