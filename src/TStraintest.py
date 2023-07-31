#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 18:34:44 2021

@author: jed-pitt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 20:02:20 2021

@author: jed-pitt
"""


import pandas as pd
import numpy as np


class TSTrainTestSplit:
    
    '''
    Functionality:
        generates tuples of train_idx, test_idx pairs that are dates.
        Assumes MultiIndex contains levels 'asset id' (or similar) and 'date'.
        Test sets created are disjoint and always of fixed size. 
        Allows for lookahead if wished (if lookahead = 0 then no lookahead).
        Contains two versions of dates-splitting:
            1. Rolling with fixed size of training set. 
            2. Recursive where the training set contains the complete past.
        
    Note: if wish to create a fixed train-test split, then just set fixed = size of test_set. 
    
    Inputs:
        train_period_length (int): size of train period, default = 84, the number of trading days in a month times 4
        
        test_period_length (int): size of test period, default = 21, the number of trading days in a month
        
        date_idx (str): name of colum in dataframe where date is present
        
        n_splits (int): number of test sets that the dataset is split; the split starts from the 
            last date towards the first date, default = 3
            use the function find_n_splits(X, test_per_length) below to find n_splits in case rolling = True (and fixed = None);
            if rolling = False (and fixed = None), the expanding training windows will automatically start from oldest date,
            but need to give the n_splits by hand
        
        lookahead (int): whether there's a gap between train and test set, default = 0
        
        rolling (boolean): whether training period length remains fixed and rolls forward with time or not, default = True
        
        fixed (None or float): if None then not a fixed split, if float, then size of test set, default = None
            note: if fixed is not None, it overrides rolling value
        
        shuffle (boolean): if True, shuffle the training dates, otherwise, keep them sequential; test dates untouched
         
    Function Outputs: 
        
        split(dataframe which has a column for date)
        returns a generator which iterates over pairs of two series, where first series contains train dates, 
        second series contains test dates
        
        get_n_splits(dataframe which has a column for date), can only be called after initialization of instance
        returns n_splits used in the intialization of instance.
        
    Remarks: 
        This is an expanded and modified version of the code in 
        https://github.com/PacktPublishing/Machine-Learning-for-Algorithmic-Trading-Second-Edition/blob/master/utils.py
       
    '''
    
    def __init__(self,
                 train_period_length = 84,
                 test_period_length = 21,
                 date_idx = 'date',
                 n_splits = 3,
                 rolling = True,
                 fixed = None, # share of test_dates in fixed
                 lookahead = 0,
                 shuffle = False):
    
        self.fixed = fixed
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.rolling = rolling
        self.shuffle = shuffle
        self.date_idx = date_idx
        
        
    def split(self, X):
        
        dat = X[self.date_idx]
        dat = dat.drop_duplicates()
        dat = dat.sort_values(ascending = False).reset_index(drop = True)
        
        T = len(dat.index)
        
        split_idx = []
        
        if self.fixed == None:
            
            for i in range(self.n_splits):
                
                test_end_idx = i*self.test_length
                test_start_idx = test_end_idx + self.test_length
                train_end_idx = test_start_idx +self.lookahead
                
                if self.rolling:
                    train_start_idx = train_end_idx + self.train_length + self.lookahead                
                else:
                    train_start_idx = T
                    
                split_idx.append([train_start_idx,train_end_idx,test_start_idx,test_end_idx])
                                    
            for train_start, train_end, test_start, test_end in reversed(split_idx):  
        
                train_dates = dat[train_end: min(train_start,T)].iloc[::-1].reset_index(drop = True)
                test_dates = dat[test_end: test_start].iloc[::-1].reset_index(drop = True)
            
                if self.shuffle:
                    train_dates = train_dates.sample(frac=1).reset_index(drop=True)
                
                yield train_dates, test_dates
                    
        else:

            test_dates = dat[0:int(self.fixed*T)].iloc[::-1].reset_index(drop = True) 
                
            train_dates = dat[(int(self.fixed*T) + self.lookahead):].iloc[::-1].reset_index(drop = True)
            
                
            if self.shuffle:
                train_dates = train_dates.sample(frac=1).reset_index(drop=True)
                
            yield train_dates, test_dates
                
            
    def get_n_splits(self):
        
        return self.n_splits
    
 
def find_n_splits(X, test_period_length, date_idx = 'date'):
    
    '''
    Finds the number of splits, if given a dataframe with unique rows

    Input:
        X (dataframe): dataframe that has as a row with the date identifier
        test_period_length (int): number of periods in one test set
        date_idx (str): name of column that has the dates
    '''
    
    X = X[date_idx].drop_duplicates()
    
    return np.floor((len(X.index)/test_period_length)).astype('int64')-1



def get_val_data(X = pd.DataFrame(), date_idx = 'date', fixed = None, n_splits = 16,test_period_length =12):
    
    '''
    Gets the validation data from a train-val dataset.
    
    Input:
        X (DataFrame): dataframe with train-val data
        Note: The date_idx variable in X needs to be sorted already at the start, whether it's an index-key or a column
        
        date_idx (str): name of date_idx, default = 'date'
        fixed (float or str or None): if float, share of val_data overall when using fixed in TSTraintestsplit
        if str, then last date before val/test date starts, if None, then pass 
        n_splits (int): number of validation periods when using rolling or expanding, default = 16
        test_period_length (int) number of dates per validation period when using rolling or expanding, default = 12
        Note: cases (fixed != None) and (n_splits, test_period_length) are mutually exclusive
        
    Output:
        Dataframes of val data and val dates.
        Note: val data DataFrame may have reset index at the end
    '''
    
    if date_idx not in list(X.columns):
        df = X.copy().reset_index()
    else:
        df = X.copy()
    
    df_dates = df[date_idx].drop_duplicates().to_frame()
    T = len(df_dates)
    
    if fixed != None:
        
        if isinstance(fixed,float):
        
            df_dates = df_dates.iloc[-int(fixed*T):,:]
        elif isinstance(fixed, str):
            df_dates = df_dates.loc[df_dates[date_idx]>fixed]
        
    else:
        t = min(T, n_splits*test_period_length)
        df_dates = df_dates.iloc[-t:,:]
        
    df = df_dates.merge(df, how = 'inner', on = date_idx)
    
    return df, df_dates.reset_index(drop = True)
        
