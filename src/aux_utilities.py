import settings

import pandas as pd, numpy as np
import os, json

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F



'''
JSON Encoder 
'''
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder,self).default(obj)
        
def create_dir_ifnexists(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        
def flatten_2dlist(lis):
    return [item for slist in lis for item in slist]

def coll_dict_to_df(dic):
    '''
    dictionary of collections of tensors to df, without column names
    '''
    lis = []
    for key, val in dic.items():
        for x in val:
            lis.append((key,x.item()))
            
    return pd.DataFrame(lis)

def subatch_sizes_to_idx(subatch_sizes):
    idx_list = [np.arange(subatch_sizes[0]).tolist()]
    for i, size in enumerate(subatch_sizes[1:],1):
        last = idx_list[-1][-1]
        idx_list.append(np.arange(last+1, last+1+size).tolist())
    return idx_list

def onehot_encoding(tens):
    '''
    Takes flattened tensor with dates, and produces mask that is used in date_splitter
    Tensor needs to be in torch.long format.
    Works for unsorted tensors as well!
    '''
    
    onehotcand = F.one_hot(tens)
    idx = onehotcand.sum(dim=0).nonzero().flatten()

    return onehotcand[:,idx]
    

def unify_date(df, frequency='monthly', date_label='date', drop_date_dec=True):
    '''
    Functionality is self-explanatory
    '''
    if 'year' not in df:
        df['year'] = df[date_label].dt.year
    if 'month' not in df:
        df['month'] = df[date_label].dt.month
    if frequency == 'monthly':
        df['day'] = 1
    elif frequency=='daily':
        if 'day' not in df:
            df['day'] = df[date_label].dt.day
    
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    if drop_date_dec:
        df.drop(columns=['year', 'month', 'day'], inplace=True)

def pcask(X, n):
    """
    Perform Principal Component Analysis (PCA) on the input data to reduce its dimensionality.

    Parameters:
        X (numpy.ndarray or pandas.DataFrame): The input data matrix, where rows represent samples and columns represent features.
        n (int): The number of principal components to retain.

    Returns:
        components (numpy.ndarray): The transformed data after PCA, where each row represents a sample, and each column represents a principal component.
        expvar (numpy.ndarray): An array containing the explained variance ratio of each retained principal component.
    """
    if isinstance(X,pd.DataFrame):
        X = X.to_numpy()
    pca = PCA(n_components = n, svd_solver='full')
    components = pca.fit_transform(X)
    expvar = pca.explained_variance_ratio_
    
    return components, expvar

def linreg_baseline(x_train, y_train, x_test, y_test, performance_measures, fit_intercept = True, scaling = False):
    
    # data preprocessing
    if isinstance(x_test, pd.DataFrame):
        idx = x_test.index
        x_train = x_train.to_numpy()
        x_test = x_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
    else:
        idx=np.arange(1,x_test.shape[0]+1,1)
        
    if len(y_test.shape)==1:
        y_test = y_test.reshape(-1,1)
        y_train = y_train.reshape(-1,1)
        
    n_targets = y_test.shape[1]
    dfindex = list(idx)*n_targets
    dfindex.sort()
    
    if scaling:
        xscaler = StandardScaler()
        yscaler = StandardScaler()
        x_train = xscaler.fit_transform(x_train).astype(np.float32)
        x_test = xscaler.transform(x_test).astype(np.float32)
        y_train = yscaler.fit_transform(y_train).astype(np.float32)
        y_test = yscaler.transform(y_test).astype(np.float32)
    
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(x_train, y_train)
    betas = model.coef_
    
    if fit_intercept:
        betas = np.concatenate([np.array(model.intercept_).reshape(-1,1),betas],axis=1)
            
    prediction = model.predict(x_test)
    if scaling:
        prediction = yscaler.inverse_transform(prediction)
    
    y_test = y_test.ravel().reshape(-1,1)
    prediction = prediction.ravel().reshape(-1,1)
    vals = np.hstack((y_test,prediction))
    
    df = pd.DataFrame(vals, columns = ['actual','prediction'], index = dfindex)
    df['residuals'] = df['actual'] - df['prediction']
    performance = [f(df['actual'], df['prediction']) for f in performance_measures]
    
    return performance, df, betas
            

def enum_dates_assets(df, date_idx = 'date', asset_idx = 'permno', save_path=f"{settings.datapath}/current_train_val_test", job_number = 15):
    '''
    Dates (date_id) will be enumerated starting from 1 on. 
    Assets (asset_id) will be enumerated from 1 on, based on their min-date (birth date) in the df.
    asset_dt_pos gives the position of the asset within a date, starting from zero
    Excess returns will have the label ret_e.
    Note that dates are always contemporaneous to returns. 
    Returns: 
    - a dataframe where first col is rix (the running index), second col is date, third col is asset_id and fourth col is 
    asset_dt_pos.
        The rest of columns stay put, including ret or ret_e.
        The dates are ranked in increasing order (with repetitions), the rix and asset_id also in increasing order.
    - dataframe with nr dates per asset
    - dataframe with nr assets per date
    '''
    for x in ['eret','e_ret']:
        if x in df.columns:
            df.rename(columns = {x:'ret_e'},inplace = True)
    if asset_idx == 'asset_id': 
        asset_idx = 'asset_idx'
        df.rename(columns ={'asset_id':asset_idx},inplace = True)
    if date_idx == 'date_id': 
        date_idx = 'date_idx'
        df.rename(columns ={'date_id':date_idx},inplace = True)    
    df = df.sort_values(by = [date_idx, asset_idx])
    df['asset_min_date'] = df.groupby(asset_idx)[date_idx].transform('min')
    small = df[[asset_idx,'asset_min_date']].sort_values(by=['asset_min_date',asset_idx]).drop_duplicates()
    small = small.reset_index().drop(columns=['index']).reset_index().rename(columns = {'index':'asset_id'})
    small['asset_id']+=1
    df = df.merge(small.drop(columns = ['asset_min_date']), how = 'inner', on = asset_idx)
    del small
    df['date_id'] = df[date_idx].rank(method='dense').astype('int')
    df['asset_dt_pos'] = df.groupby('date_id').cumcount()
    df.drop(columns = ['asset_min_date'],inplace = True)
    
    df_othercols = [col for col in df.columns if col not in ['date_id','asset_id','ret_e','ret','asset_dt_pos', asset_idx, date_idx]]
    main_cols = [col for col in ['date_id','asset_id', 'asset_dt_pos',asset_idx, date_idx] if col in df.columns]
    for col in ['ret','ret_e']:
        if col in df.columns:
            main_cols.append(col)
    df = df[main_cols+df_othercols].sort_values(by = ['date_id','asset_id'])
    df = df.reset_index(drop=True).reset_index().rename(columns = {'index':'rix'})
    
    if save_path is not None:
        enum_dates = df[[date_idx,'date_id']].drop_duplicates()
        enum_assets = df[[asset_idx,'asset_id','date_id','asset_dt_pos']].drop_duplicates()
        try:
            enum_dates.to_excel(f"{save_path}/enum_dates_{job_number}.xlsx")
            enum_assets.to_excel(f"{save_path}/enum_assets_{job_number}.xlsx")
        except:
            enum_dates.to_pickle(f"{save_path}/enum_dates_{job_number}.pkl")
            enum_assets.to_pickle(f"{save_path}/enum_assets_{job_number}.pkl")
        
    dates_per_asset = df[['date_id','asset_id']].drop_duplicates().groupby('asset_id').size().reset_index().sort_values('asset_id')
    dates_per_asset.rename(columns = {0:'date_counts'},inplace = True)
    assets_per_date = df[['date_id','asset_id']].drop_duplicates().groupby('date_id').size().reset_index().sort_values('date_id')
    assets_per_date.rename(columns = {0:'asset_counts'}, inplace = True)

    return df, dates_per_asset, assets_per_date

def asset_dt_pos_bench(ixm):
   
    '''
    For unbalanced dataset, this produces the last column in the ixmap file, which gives the position of the current asset 
    in the previous date, with -1 for assets not present in previous date.
    Used to fetch the right assets in benchmark_cov (which are saved with one date ahead). 
    '''
    ixm = ixm.sort_values(['date_id','asset_id'])
    orig_len = ixm.shape[0]
    assets = list(ixm.loc[ixm['date_id']==1, 'asset_id'])
    result = [np.array([np.ones(len(assets)), assets, -1*np.ones(len(assets))]).T]
    for date in ixm.date_id.unique()[1:]:
        assets = list(ixm.loc[ixm['date_id']==date, 'asset_id'])
        previous={}
        previous = dict(zip(list(ixm.loc[ixm['date_id']==date-1, 'asset_id']),range(len(ixm.loc[ixm['date_id']==date-1,'asset_id']))))
        current = dict(zip(assets,-1*np.ones(len(assets))))
        for asset in assets:
            if asset in previous.keys():
                current[asset]=previous[asset]
                
        result.append(np.array([date*np.ones(len(assets)),assets,list(current.keys())]).T)
        
    result = pd.DataFrame(np.concatenate(result,axis=0), columns = ['date_id','asset_id', 'asset_dt_pos_bench'])
    ixm = ixm.merge(result, how = 'inner', on = ['date_id','asset_id'])
    assert len(ixm)==orig_len
    
    return ixm

def raise_naninf(df, split_col=None):
    '''
    Functionality is self-explanatory; split_col is for large dataframes where we can hit stack memory bounds
    '''
    checknaninf = 0
    if split_col is None:
        checknaninf = df.isin([np.inf, -np.inf]).sum().sum() or df.isnull().sum().sum()
        if checknaninf != 0:
            raise ValueError('Dataframe has nans or infs!')
    else:
        for item in df[split_col].drop_duplicates().unique():
            dfloc = df[df[split_col] == item]
            checknaninf = dfloc.isin([np.inf, -np.inf]).sum().sum() or dfloc.isnull().sum().sum()
            if checknaninf != 0:
                raise ValueError(f'Dataframe has nans or infs when {split_col}={item}!')
    print('All is good. Dataframe has no nans or infs.')


def cumret(df, start, end, ret_label='retadj', date_label='date', asset_label='permno', cumret_label='ltrev'):
    '''
    Utility to calculate cumret from a start date to an end date. This is the holding period return from start date to end date.
    df requires columns with date, asset, and return
    returns full dataframe
    
    to calculate from time t to time t+h, need start = t and end = t+h-1 
    
    '''
    df.sort_values(by=[asset_label, date_label], inplace=True)
    df['temp'] = df[ret_label] + 1
    df['cumret'] = df.groupby([asset_label])['temp'].cumprod()
    df[cumret_label] = (df.groupby([asset_label])['cumret'].shift(start) / df.groupby([asset_label])['cumret'].shift(
        end)) - 1
    df.drop(columns=['temp'], inplace=True)

    return df

def signal_constructor(data_input, cols, date_id='date', frequency='monthly', asset_id='permno', normalization='std'):
    '''
        Goal: create zero-cost long-short portfolio by ranking according to one signal. Do this for multiple signals.        
        allows normalization of demeaned signal by std, or by L1 norm
        Note: leave date_id = None if already have columns ['year'], ['year', 'month'] or ['year','month','day'], otherwise it produces them  
    '''

    nr_assets = {}
    nr_avail_signals = {}
    cols_aux = list(cols)
    cols = list(cols)
    data_input.reset_index(inplace=True)

    if 'id' in data_input.columns:
        data_input.drop(columns=['id'], inplace=True)
    data_input.rename(columns={"index": "id"}, inplace=True)

    date_dec = ['year']
    if date_id is not None:
        data_input['year'] = data_input[date_id].dt.year

    if frequency == 'monthly':
        date_dec.append('month')
        if date_id is not None: data_input['month'] = data_input[date_id].dt.month
    elif frequency == 'daily':
        date_dec.append('month')
        date_dec.append('day')
        if date_id is not None: 
            data_input['month'] = data_input[date_id].dt.month
            data_input['day'] = data_input[date_id].dt.day

    time_dates_assets = data_input[[*date_dec, asset_id]].drop_duplicates().sort_values(by=[*date_dec, asset_id]).copy()
    data = pd.DataFrame([])

    print("Signal construction\n")

    for idx, time in time_dates_assets[date_dec].drop_duplicates().iterrows():
        if frequency == 'monthly':
            n_assets = time_dates_assets.loc[(time_dates_assets["year"] == time["year"]) & (
                        time_dates_assets["month"] == time["month"]), asset_id].nunique()
            nr_assets[(time['year'], time['month'])] = n_assets
            ex = data_input.loc[
                (data_input["year"] == time["year"]) & (data_input["month"] == time["month"]), [*cols_aux, "id",
                                                                                                asset_id]].copy()
            print("Year:", time["year"], "Month:", time["month"], "nr assets:", n_assets)
        elif frequency == 'daily':
            n_assets = time_dates_assets.loc[
                (time_dates_assets["year"] == time["year"]) & (time_dates_assets["month"] == time["month"]) & (
                            time_dates_assets["day"] == time["day"])
                , asset_id].nunique()
            nr_assets[(time['year'], time['month'], time['day'])] = n_assets
            ex = data_input.loc[(data_input["year"] == time["year"]) & (data_input["month"] == time["month"]) & (
                        data_input["day"] == time["day"]), [*cols_aux, "id", asset_id]].copy()
            print("Year:", time["year"], "Month:", time["month"], "Day:", time['day'], "nr assets:", n_assets)
        else:
            n_assets = time_dates_assets.loc[(time_dates_assets["year"] == time["year"]), asset_id].nunique()
            nr_assets[(time['year'])] = n_assets
            ex = data_input.loc[(data_input["year"] == time["year"]), [*cols_aux, "id", asset_id]].copy()
            print("Year:", time["year"], "nr assets:", n_assets)

        if len(ex) > 0:
            trans = ex[cols_aux].rank(axis=0, pct=False, na_option='keep', method='average')
            cols = [f'{col}_weight' for col in cols_aux]
            trans = pd.DataFrame(trans.to_numpy(), columns=cols, index=ex['id'])
            trans[asset_id] = list(ex[asset_id])
            trans.sort_values(by=asset_id, inplace=True)
            for ix in date_dec:
                trans[ix] = time[ix]
            '''important for processing case of nans below'''
            # separate nans
            if trans[cols].isnull().sum().sum() == 0:
                print("No nans.")
                if normalization == 'std':

                    scaler = StandardScaler()
                    trans[cols] = scaler.fit_transform(trans[cols].to_numpy() / (n_assets + 1))

                elif normalization == 'l1':
                    trans[cols] = trans[cols] / (n_assets + 1)
                    trans[cols] = (trans[cols] - trans[cols].mean(axis=0))
                    for col in cols:
                        trans[f"{col}_norm_1"] = np.linalg.norm(trans[col], ord=1)
                        trans[col]/=trans[f"{col}_norm_1"]
                    trans.drop(columns = [f"{col}_norm_1" for col in cols], inplace = True)
                    
                trans.fillna(value=0,inplace=True)

            else:
                if frequency=='monthly':
                    print("Year:", time["year"], "Month:", time["month"], f"has {trans.isnull().sum().sum()} nans")
                elif frequency=='daily':
                    print("Year:", time["year"], "Month:", time["month"], "Day:",time["day"],f"has {trans.isnull().sum().sum()} nans")
                print("Dealing with nans.")
                for col in cols:
                    transaux = trans[[asset_id, col]].copy()
                    nulls = transaux[transaux[col].isna()]
                    ok = transaux[~transaux[col].isna()]
                    nr_assets_signal = ok[asset_id].nunique()
                    nr_avail_signals[(*[time[ix] for ix in date_dec], col)] = nr_assets_signal
                    if len(ok) > 1:
                        if normalization == 'std':
                            scaler = StandardScaler()
                            ok[col] = scaler.fit_transform(ok[[col]].to_numpy() / (nr_assets_signal + 1))
                        elif normalization == 'l1':
                            ok[col] = ok[col] / (nr_assets_signal + 1)
                            ok[col] = (ok[col] - ok[col].mean(axis=0)) 
                            ok[f"{col}_norm_1"] = np.linalg.norm(ok[col], ord=1)
                            ok[col]/=ok[f"{col}_norm_1"]
                            ok.drop(columns = [f"{col}_norm_1"], inplace = True)
                    elif len(ok) == 1:
                        print("ok has len=1!")
                        '''remember, it needs to be a zero cost portfolio'''
                        ok[col] = 0
                        
                    nulls.fillna(value=0, inplace=True)
                    ok.fillna(value=0, inplace=True)
                    transaux1 = pd.concat([ok, nulls], axis=0).sort_values(by=[asset_id])
                    assert (len(transaux) == len(transaux1))
                    trans[col] = transaux1[col]

            data = pd.concat([data, trans], axis=0)
            
            if data.isnull().sum().sum():
                print(data.isnull().sum()>0)
        else:
            print(f"Extract has length {len(ex)}")
            trans = pd.DataFrame()
    
    nr_assets = pd.DataFrame.from_dict(nr_assets, orient='index', columns=['nr_total_assets']).reset_index()
    nr_assets['year'] = nr_assets['index'].apply(lambda x: int(str(x).split(',')[0][1:]))

    if frequency == 'monthly':
        nr_assets['month'] = nr_assets['index'].apply(lambda x: int(str(x).split(',')[1][:-1]))
    elif frequency == 'daily':
        nr_assets['month'] = nr_assets['index'].apply(lambda x: int(str(x).split(',')[1]))
        nr_assets['day'] = nr_assets['index'].apply(lambda x: int(str(x).split(',')[2][:-1]))
    nr_assets.drop(columns=['index'], inplace=True)

    del trans, time_dates_assets
    
    if len(nr_avail_signals) > 0:
        nr_avail_signals = pd.DataFrame.from_dict(nr_avail_signals, orient='index',
                                                  columns=['nr_assets_signal']).reset_index()
        nr_avail_signals['year'] = nr_avail_signals['index'].apply(lambda x: int(str(x).replace("(","").split(',')[0]))
        if frequency == 'monthly':
            nr_avail_signals['month'] = nr_avail_signals['index'].apply(lambda x: int(str(x).split(',')[1]))
        if frequency == 'daily':
            print(nr_avail_signals['index'])
            nr_avail_signals['month'] = nr_avail_signals['index'].apply(lambda x: int(str(x).split(',')[1]))
            nr_avail_signals['day'] = nr_avail_signals['index'].apply(lambda x: int(str(x).split(',')[2]))
        nr_avail_signals['signal'] = nr_avail_signals['index'].apply(lambda x: str(x).split(',')[2][:-1])
        nr_avail_signals['signal'] = 'nr_' + nr_avail_signals['signal']  # kill the _weight
        nr_avail_signals['signal'] = nr_avail_signals['signal'].apply(lambda x: x.replace("'", ''))
        nr_avail_signals['signal'] = nr_avail_signals['signal'].apply(lambda x: x.replace(" ", ''))
        nr_avail_signals.drop(columns=['index'], inplace=True)
        if frequency=='monthly':
            idx = ['year', 'month']
        elif frequency=='daily':
            ['year', 'month','day']
        nr_avail_signals = nr_avail_signals.pivot(index=idx, columns='signal',
                                                  values='nr_assets_signal').reset_index()
        assert (nr_avail_signals.isnull().sum().sum() == 0)

        nr_assets = nr_assets.merge(nr_avail_signals, how='inner', on=date_dec)
        del nr_avail_signals
        
    raise_naninf(data)
    
    print("\nDone with signal construction")
    
    return data.sort_values(by=[*date_dec, asset_id]).set_index(keys=[*date_dec, asset_id]).reset_index(), nr_assets

# use portfolio weight to compute return
def compute_port_return(df, ret_col = 'ret'):
    weight_cols = [col for col in df if col.endswith('weight')]
    r_fun = lambda x: pd.DataFrame((np.expand_dims(x[ret_col], axis=0)) @ x[weight_cols])
    port_return = df.groupby('date').apply(r_fun)
    port_return.reset_index(inplace=True)
    port_return.drop(columns='level_1', inplace=True)  
    ret_cols = port_return.columns.str.replace('weight', 'return')
    port_return.columns = ret_cols

    return port_return

def df_lags_builder(df, seq_len):
    
    '''
    Takes a dataframe that has date in its index, and produces seq_len lags of it. Note: seq_len > 0 here. 
    Returns: a dataframe with lagged values from past to present (read left to right), that ends with the values from df 
    (i.e. ends contemporaneous values)
    Apply torch.view(N, seq_len, df.shape[1]) in forward method of RNN, to get it ready for RNN
    Note: for RNN and models which do RNN merge, during preprocessing, the sequences should be in the very last columns of the X_train, X_test datasets
    '''
    df_1 = df.copy()
    df_1 = df_1.sort_index()
    df_final = df.copy()
    for i in range(1, seq_len+1):
        df_aux = df_1.shift(i)
        df_aux.rename(columns = {col:col+f'_lag_{i}' for col in df_1.columns}, inplace = True)
        df_final = pd.concat([df_aux, df_final],axis = 1)
   
    return df_final.dropna()

'''
    Utilities for selecting from covariance based on dates or blocks
'''

def create_idx_list(shapes_tuple):
    idx = torch.arange(sum(shapes_tuple)).to(torch.long)
    return torch.split(idx, shapes_tuple)
