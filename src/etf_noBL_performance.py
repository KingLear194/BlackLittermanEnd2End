import numpy as np, pandas as pd, glob
import cvxpy as cp

import performancemeasures as pm
import settings

def mvpo_weights(keys, means: np.ndarray, ret_covs: np.ndarray, risk_av: float = 5, tcost: float = 0.005, lev_constraint = 0.5) -> np.ndarray:
    """
    computes next H period portfolio weights given mean and covariance forecasts.
    :param means: H x number of asset, mean forecast H steps ahead
    :param ret_covs: H x n x n, covariance forecast H steps ahead
    :param risk_av: risk aversion parameter
    :param tcost: transaction cost scaling
    :return: an array of H x n of weights
    """
    keys.sort()
    T, N = means.shape
    assert(T == len(keys))
    weights = {}
    w = cp.Variable((N,1))
   
    objective = means[0] @ w - risk_av / 2 * cp.quad_form(w, ret_covs[0])
    constr = [cp.norm(w,2) <= lev_constraint]
    prob = cp.Problem(cp.Maximize(objective), constr)
    prob.solve()
    weights[keys[0]] = w.value
    for t in range(1, T):
        w = cp.Variable((N,1))
        objective = means[t] @ w - risk_av / 2 * cp.quad_form(w, ret_covs[t]) - tcost * cp.norm(w - weights[keys[t-1]], 1)
        constr = [cp.norm(w,2) <= lev_constraint]
        prob = cp.Problem(cp.Maximize(objective), constr)
        prob.solve()
        weights[keys[t]] = w.value
    
    return weights

def main(nr_PCA_facs, savepath, project_name = 'etf'):
    
    output = pd.read_pickle(glob.glob(f"{savepath}pf_output*pkl")[0]).sort_values(['date','asset_idx'])
    enum_dates = pd.read_excel(glob.glob(f"{savepath}*enum_dates*xlsx")[0]).drop(columns = ['Unnamed: 0'])
    
    output = output.merge(enum_dates, how = 'inner', on = 'date')[['date','date_id',
        'phase','asset_idx','ret_e','pred_ret_e_mean','drawdown','leverage']].drop_duplicates()
    mu_path = f"{settings.finalpath}/{project_name}_b_{nr_PCA_facs}fac/*mu.npy"
    mu_files = glob.glob(mu_path)
    mu_dict, cov_dict = {}, {}
    
    for file in mu_files:
        key = int(file.replace(f"{settings.finalpath}/{project_name}_b_{nr_PCA_facs}fac/","").replace("/","").replace(f"{project_name}_b_","").replace("_mu.npy","").replace("etf_b_","").replace("_mu.npy",""))
        mu_dict[key] = np.load(file)
        cov_dict[key] = np.load(file.replace("mu","cov"))
    
    keys= list(mu_dict.keys())
    keys.sort()
    
    means = np.array([mu_dict[key].flatten() for key in keys])
    ret_covs = np.array([cov_dict[key] for key in keys])
        
    weights = mvpo_weights(keys, means,ret_covs)
    
    wei = [np.concatenate((np.array([date]*14).reshape(-1,1), weights[date].reshape(-1,1)), axis = 1) for date in weights.keys()]
    wei = pd.DataFrame(np.concatenate(wei, axis = 0), columns = ['date_id','mvpo_weights'])
    assets = list(output.asset_idx.unique())*(int)(wei.shape[0]/14)
    wei['asset_idx'] = np.array(assets)
    wei = pd.DataFrame(wei, columns = ['date_id','asset_idx','mvpo_weights'])
    wei['date_id'] = wei['date_id'].astype(int)
    wei['mvpo_leverage'] = wei.groupby('date_id')['mvpo_weights'].transform(lambda x: x.abs().sum())
    
    ret_e = output[['date','date_id','phase','asset_idx','ret_e']].drop_duplicates()
    if project_name=='etf':
        ret_e = ret_e.loc[(ret_e['date_id']>=4)&(ret_e['date_id']<=176),:]
    elif project_name == 'etf_daily':
        ret_e = ret_e.loc[ret_e['date']<='2019-12-31',:]
    
    rets = {}
    for date in weights.keys():
        ret = np.array(ret_e.loc[ret_e['date_id']==date, 'ret_e']).flatten()
        #print(len(ret))
        if len(ret)==14:
            rets[date] = np.dot(ret,weights[date])
    
    rets = pd.DataFrame.from_dict(rets, orient = 'index',columns = ['mvpo_ret']).reset_index().rename(columns = {'index':'date_id'})
    ret_e = ret_e.merge(rets, how = 'inner', on = 'date_id')[['date','date_id','phase','mvpo_ret']].drop_duplicates()
    
    perf = pm.produce_port_statistics(ret_e, ret_label = 'mvpo_ret')
    print(f"Performance of benchmark is as follows:\n{perf}")
    
    ret_e = pm.get_drawdown_series(ret_e, trainend = None, ret_label = 'mvpo_ret').rename(columns = {'drawdown':'mvpo_drawdown'})[['date_id','mvpo_ret','mvpo_drawdown']]
    ret_e = output.merge(ret_e, how = 'inner', on = ['date_id'])
    
    ret_e = ret_e.merge(wei, how = 'inner', on = ['date_id','asset_idx'])
    ret_e = ret_e.drop_duplicates()
    
    # saving
    ret_e.to_pickle(savepath+ f"NoBLBenchmark/pf_comparison_BL_vs_mvpo_{nr_PCA_facs}fac.pkl")
    ret_e.to_excel(savepath+f"NoBLBenchmark/pf_comparison_BL_vs_mvpo_{nr_PCA_facs}fac.xlsx")
    perf.to_pickle(savepath+f"NoBLBenchmark/pf_performance_mvpo_{nr_PCA_facs}fac.pkl")
    perf.to_excel(savepath+f"NoBLBenchmark/pf_performance_mvpo_{nr_PCA_facs}fac.xlsx")
    
if __name__ == '__main__':
    
    savepath = f'{settings.trainpath}/etf/Results/June27_monthly/'
    project_name = 'etf'
    main(3, savepath, project_name)
    main(5, savepath, project_name)
    main(7, savepath, project_name)
    main(10, savepath, project_name)
    
    
    