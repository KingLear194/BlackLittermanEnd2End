# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:52:55 2022

@author: jetlir
"""

import numpy as np
from aux_utilities import pcask, linreg_baseline
from sklearn.metrics import r2_score, mean_squared_error


class RollingPCAFactors():
    
    '''
    Use linreg_baseline function to predict the value... and pcask to do the PCA 
    
    We have the model 
    
        Y = alpha + beta x Factors + u
        
    Where the Factors are estimated via PCA from raw_factors
    
    date_id from Y dataframe needs to be the key for the dict in raw_factors, and the mapping from date_id in Y to keys in raw_factors_dict needs to be onto.
    
    Y variable is of shape (nr_dates, nr_assets+1). It needs to have column with date_id, and also the asset_ids in the other columns. The order of the asset_ids
    in the other columns needs to match the order of the asset_id in the raw factors. The dates need to be ordered. 
    
    Here, since using enum_dates_assets for both Y and also the covariance production, the ordering of the dates is ensured. 
    
    calc_rolling_factor_regressions returns predictions for one-step-ahead, coefficients and last_fitted_y
    '''
    
    def __init__(self, Y, raw_factors_dict, nr_factors, window = 36, skip_first = False, date_id = 'date_id', asset_id = 'asset_id', label_col = 'ret'):
        super(RollingPCAFactors, self).__init__()
        
        self.Y = Y # these are the returns in dataframe form
        self.raw_factors = raw_factors_dict # this is going to be a dict of covariances with a covariance for each date, from which we calculate the PCA factors, the date needs to be the same as in the Y dataframe!
        self.nr_factors = nr_factors # this is going to be used for the pca
        self.label_col = label_col
        self.window = window
        self.date_id = date_id
        self.asset_id = asset_id
        self.skip_first = skip_first # whether to start estimation only after window dates are available... if not then start with at least two periods and do 
        # expanding windows until the window size becomes equal to what is needed for rolling
        
        self.factors = {}
        for key, val in self.raw_factors.items():
            self.factors[key] = pcask(self.raw_factors[key], self.nr_factors)[0]
            
        self.predictions, self.coeff, self.last_fitted_y = {}, {}, {}
        
    def calc_rolling_factor_regressions(self):
        
        self.dates = self.Y[self.date_id].unique()
        nr_assets = self.Y[self.asset_id].nunique()
        if min(list(self.factors.keys()))>self.dates.min():
            self.dates = self.dates[1:]
        
        for ix, current_date in enumerate(self.dates[1:-1],2): # don't estimate in the first and in the last date
            
            if not self.skip_first:
                self.wind = min(self.window,ix)
            else:
                self.wind = self.window
                if ix<self.window:
                    continue
                
            past_dates = self.dates[ix-self.wind:ix]
            next_date = self.dates[ix]
            y_train = self.Y.loc[self.Y[self.date_id].isin(past_dates),self.label_col].to_numpy()
            x_train = np.concatenate([self.factors[date] for date in past_dates], axis = 0)
            y_test = self.Y.loc[self.Y[self.date_id].isin([past_dates[-1],next_date]),self.label_col].to_numpy()
            x_test = np.concatenate([self.factors[past_dates[-1]],self.factors[next_date]], axis = 0)
            
            performance, df, coeff = linreg_baseline(x_train, y_train, x_test, y_test, [r2_score, mean_squared_error], fit_intercept = True, scaling = False)
            
            self.last_fitted_y[current_date] = df['prediction'].to_numpy()[:nr_assets]
            self.predictions[next_date] = df['prediction'].to_numpy()[nr_assets:]
            self.coeff[current_date] = coeff
            
    def get_benchmark(self):
        
        '''
        saving them one date_id up, i.e. for the date_id where they are used to predict (measurable w.r.t. one date less than what is outputted)
        '''
        
        benchmark_dic = {}
        start = np.where(self.dates == min(list(self.last_fitted_y.keys())))[0][0]+1
        
        for ix, date in enumerate(self.dates[start:],start):
            benchmark_dic[date] = [self.last_fitted_y[self.dates[ix-1]],self.raw_factors[self.dates[ix-1]]]    
        
        return benchmark_dic
           



