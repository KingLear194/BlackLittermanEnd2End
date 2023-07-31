#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 22:59:08 2022

@author: jed169

Calculate benchmarks here CAPM and for the rest.
"""

import os
import torch, pandas as pd, numpy as np, glob
import settings, aux_utilities, benchmark


def etf_benchmark(nr_factors = 3, project = 'etf'):
    
    '''
    the date of the covs and mu-s when saving is the date where the benchmark can be used.
    This makes sure that the +1 step in the trainer.py when collecting the returns to evaluate performance is correct!
    '''
    
    os.chdir(settings.finalpath+f'/{project}_cov')
    filenameslist = glob.glob("*.npy")
    filenameslist.sort()
    
    cov_dict = {}
    for filename in filenameslist:
        key = int(filename.replace(project,"").replace('_','').replace('cov','').replace(".npy",""))
        cov_dict[key] = np.load(filename)
    
    etfdf = pd.read_pickle(settings.finalpath+f"{project}_returns.pkl")
    Y, _, _ = aux_utilities.enum_dates_assets(etfdf, date_idx = 'date', asset_idx ='asset_id', save_path=None)
    
    Y = Y[['date_id','asset_id','ret_e']]
    etf_benchmark = benchmark.RollingPCAFactors(Y, raw_factors_dict = cov_dict, nr_factors = nr_factors, window = 10,
                                      date_id = 'date_id', asset_id = 'asset_id', label_col = 'ret_e', skip_first = False)

    etf_benchmark.calc_rolling_factor_regressions()
    bench = etf_benchmark.get_benchmark()
    
    savepath = f"{settings.finalpath}/{project}_b_{nr_factors}fac"
    aux_utilities.create_dir_ifnexists(savepath)
    
    for key, value in bench.items():
        with open(f"{savepath}/{project}_b_{key}_mu.npy","wb") as f:
            np.save(f, value[0])
        with open(f"{savepath}/{project}_b_{key}_cov.npy","wb") as ff:
            np.save(ff, value[1])
            

def check_invertibility_etf(project = 'etf'):
    
    path = settings.finalpath+f'{project}_cov/'
    lis = []
    for i in range(2,177):
        
        cov = np.load(path+f'{project}{i}_cov.npy')
        cov = torch.Tensor(cov)
        try:
            _ = torch.linalg.inv(cov)
        except np.linalg.LinAlgError as err:
            print(str(err), " error in inverting covariance matrix in period ", i)
            lis.append(i)
            
    return lis

if __name__ == '__main__':
    
   
   print(check_invertibility_etf('etf'))
   
   for nrfac in [3,5,7,10]:
       etf_benchmark(nr_factors = nrfac, project = 'etf')
  
   
