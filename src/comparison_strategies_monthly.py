"""
Implements basic strategies to compare with main results for etfs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from aux_utilities import cumret
from performancemeasures import mdd
from torch import tensor
import matplotlib
from tabulate import tabulate
from performancemeasures import analyze_returns

# matplotlib
matplotlib.rcParams.update({"axes.grid": False})
matplotlib.rcParams.update({'font.weight': 'bold'})
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

from settings import cov_path

def risk_parity_weights(ret_cov):
    n = ret_cov.shape[0]
    weights = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.quad_form(weights, ret_cov) - cp.sum(cp.log(weights)) / n)
    prob = cp.Problem(objective)
    prob.solve()
    return np.array(weights.value / np.sum(weights.value), dtype=np.float64)

def compute_rp_weights(input_df, covariance_path=cov_path):
    # initialize with equal weights
    df = input_df.copy()
    df["rp_weights"] = 1/14
    date_ids = df["date_id"].unique()
    for ct, ix in enumerate(date_ids[:-1]):
        cov = np.load(covariance_path+f"etf{ix}_cov.npy")
        df.loc[df.date_id == date_ids[ct+1], "rp_weights"] = risk_parity_weights(cov)
    return df

def compute_momentum_rot_weights(input_df):
    # sector
    df = input_df.copy()
    df["mom_rot_weights"] = 0
    cumret(df, start=1, end=12, ret_label= "ret_e", date_label="date", cumret_label="mom12", asset_label="asset_idx")
    df["mom12"] = df["mom12"].fillna(0)
    asset_list = [a for a in df["asset_idx"].unique() if a.startswith("X")]
    assign_w = np.linspace(0, 1, num=len(asset_list))
    assign_w = assign_w / np.sum(assign_w)
    df.loc[df["asset_idx"].isin(asset_list), "rank"] = df.loc[df["asset_idx"].isin(asset_list)].groupby("date")["mom12"].rank("dense", ascending=True).astype(int)
    df.loc[df["asset_idx"].isin(asset_list), "mom_rot_weights"] = assign_w[df.loc[df["asset_idx"].isin(asset_list),"rank"].to_numpy(dtype=int)-1]

    return df

def compute_60_40(input_df):
    df = input_df.copy()
    df["60_40_weights"] = 0
    df.loc[df.asset_idx == "SPY", "60_40_weights"] = 0.6
    df.loc[df.asset_idx == "TLT", "60_40_weights"] = 0.4
    return df

def compute_equal_weights(input_df, col_label = "equal_weights"):
    df = input_df.copy()
    df[col_label] = 1/14
    return df

def compute_momentum_rev_weights(input_df):
    df = input_df.copy()
    df["mom_rev"] = 1
    df["mom_rev_weights"] = 0
    assign_w = np.linspace(0, 1, num=df.asset_idx.nunique())
    assign_w = assign_w / np.sum(assign_w)
    cumret(df, start=1, end=3, ret_label="ret_e", date_label="date", cumret_label="mom_rev", asset_label="asset_idx")
    df["mom_rev"] = -df["mom_rev"]
    df["rank"] = df.groupby("date")["mom_rev"].rank("dense", ascending=True)
    df = df.fillna(1)
    df["mom_rev_weights"] = assign_w[df["rank"].to_numpy(dtype=int)-1]
    return df


def compute_port_return(input_df, ret_col = 'ret', weight_cols=["rp_weights"]):
    df = input_df.copy()
    r_fun = lambda x: pd.DataFrame((np.expand_dims(x[ret_col], axis=0)) @ x[weight_cols])
    port_return = df.groupby('date').apply(r_fun)
    port_return.reset_index(inplace=True)
    port_return.drop(columns='level_1', inplace=True)  # not sure why this column shows up
    ret_cols = port_return.columns.str.replace('weights', 'returns')
    port_return.columns = ret_cols
    return port_return


def plot_portfolios(df, ret_columns=["returns"], date_column = "date", portfolio_name_dict={"column_name": "portfolio_name"}):
    plt_df = df.copy()
    # Append zeros so that all plots start at 1
    df_zero = pd.DataFrame(0, index=[0], columns=plt_df.columns)
    plt_df =  pd.concat([df_zero, df]).reset_index(drop=True)
    plt_df.loc[plt_df[date_column]==0, date_column] = df.date.min()
    plt_df[date_column] = pd.to_datetime(plt_df[date_column])
    fig, ax = plt.subplots()
    ax.plot(plt_df[date_column], (plt_df[ret_columns]+1).cumprod(), label = [portfolio_name_dict[c[:-8]] for c in ret_columns])
    if len(ret_columns) > 1:
        ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Value Index")
    return fig, ax

def compute_statistics_df(df, ret_columns=["returns"], portfolio_name_dict={"column_name": "portfolio_name"}):
    stats_df = pd.DataFrame(index=[portfolio_name_dict[c[:-8]] for c in ret_columns], columns=["sharpe", "sortino", "calmar", "mdd"])
    ratios = ["sharpe", "sortino", "calmar"]
    for port in ret_columns:
        stats_df.loc[portfolio_name_dict[port[:-8]], ratios] = analyze_returns(df[port]).to_numpy().squeeze()
        stats_df.loc[portfolio_name_dict[port[:-8]], "mdd"]= mdd(tensor(df[port].to_numpy()))
    return stats_df


if __name__ == "__main__":
    
    # change paths accordingly
    
    # Where the benchmark is read from
    benchmark_path = "D:\Dropbox (Princeton)\BL_model\Data\Results\June27_etf_monthly/NoBLBenchmark/"
    # results_path = "D:\Dropbox (Princeton)\BL_model\Data\Results\June27_etf_monthly/"

    # Overleaf's Dropbox figure folder
    figure_path = "D:\Dropbox (Princeton)\Apps\Overleaf\Black-LittermanEnd-To-End/figures/"

    # Overleaf's Dropbox portfolio statistics folder
    table_path = "D:\Dropbox (Princeton)\Apps\Overleaf\Black-LittermanEnd-To-End/portfolio_stats/"

    # folder containing ETF returns
    return_path = "D:\Dropbox (Princeton)\BL_model\Data\Results\June27_etf_monthly/"

    # etf blend2end return path, including portfolio weights
    return_df_path = return_path + "pf_output_jnr_363_bs_28_lr_0.001_l2reg_none_l1reg_none__mean_ensembled.pkl"

    # benchmark weights path
    bm_weights_path = f"{benchmark_path}pf_comparison_BL_vs_mvpo_5fac.pkl"

    # benchmark performance path
    bm_performance_path = f"{benchmark_path}"

    bl_performance_path = return_path + "pf_performance_jnr_363_bs_28_lr_0.001_l2reg_none_l1reg_none__mean_ensembled.pkl"

    bl_return_path = return_path + "pf_returns_jnr_363_bs_28_lr_0.001_l2reg_none_l1reg_none__mean_ensembled.pkl"

    ret_df = pd.read_pickle(return_df_path)

    portfolio_name_dict = {"BLEnd2End": "BLEnd2End", "equal": "Equal Weights", "rp": "Risk Parity", "yc": "Treasury Yield Curve",
                           "BLEnd2End_sec": "BLEnd2End Sectors", "BLEnd2End_t": "BLEnd2End Treasury",
                           "mom_rot": "Sector Momentum Rotation", "mom_rev": "Sector Momentum Reversal", "60_40": "60-40",
                           "benchmark_5fac": "5-Factor Benchmark",
                           "benchmark_7fac": "7-Factor Benchmark",
                           "benchmark_10fac": "10-Factor Benchmark",
                           "benchmark_3fac": "3-Factor Benchmark"}

    ret_df = ret_df.rename(columns={"weights_mean": "BLEnd2End_weights"})

    # get date_id and date correspondence
    date_to_id = pd.read_excel(return_path + "enum_dates_363.xlsx", index_col=0)
    # merge date_id with return df
    # return df contains the return of all assets.
    ret_df = pd.merge(ret_df, date_to_id, on="date", how="left")

    # sector list contains all sector tickers, which are the ETF tickers that start with "X"
    sec_list = [c for c in ret_df["asset_idx"].unique() if c.startswith("X")]
    sec_df = ret_df.loc[ret_df["asset_idx"].isin(sec_list)]

    ret_df = compute_rp_weights(ret_df)
    ret_df = compute_60_40(ret_df)
    ret_df = compute_equal_weights(ret_df, col_label="equal_weights")

    # benchmark bm
    fac5_df = pd.read_pickle(bm_weights_path)

    bms = fac5_df[["date","phase", "asset_idx", "mvpo_weights", "ret_e"]].rename(columns={"mvpo_weights": "benchmark_5fac_weights"})
    bms["BLEnd2End_weights"] = ret_df["BLEnd2End_weights"]

    portfolio_bms = compute_port_return(bms, ret_col="ret_e", weight_cols = [c for c in bms if c.endswith("weights")])
    portfolio_bms = portfolio_bms[['date','BLEnd2End_returns', 'benchmark_5fac_returns'
       ]]

    test_bms = bms.loc[bms["phase"] == "test"]

    portfolio_test_bms = compute_port_return(test_bms, ret_col="ret_e", weight_cols = [c for c in bms if c.endswith("weights")])
    portfolio_test_bms = portfolio_test_bms[['date','BLEnd2End_returns', 'benchmark_5fac_returns'
       ]]

    ret_df = ret_df.loc[ret_df["phase"]=="test"]

    sec_df = compute_momentum_rot_weights(sec_df)
    sec_df = compute_momentum_rev_weights(sec_df)
    sec_df = sec_df.rename(columns={"BLEnd2End_weights": "BLEnd2End_sec_weights"})
    sec_df = sec_df.loc[sec_df["phase"]=="test"]


# compute portfolio returns
    portfolio_all = compute_port_return(ret_df, ret_col = "ret_e", weight_cols = [c for c in ret_df if c.endswith("weights")])
    portfolio_sec = compute_port_return(sec_df, ret_col = "ret_e", weight_cols = [c for c in sec_df if c.endswith("weights")])

    fig, ax = plot_portfolios(portfolio_all, ret_columns=[c for c in portfolio_all if c.endswith("returns")], portfolio_name_dict=portfolio_name_dict)
    ax.plot(ret_df.date.unique(), (ret_df.loc[ret_df["asset_idx"]=="SPY", "ret_e"]+1).cumprod(), label="SPY")
    ax.set_title("Portfolio Performance on Test Set")
    fig.savefig(figure_path+"etf_all_assets.png")
    plt.show()

    fig, ax = plot_portfolios(portfolio_sec,  ret_columns=[c for c in portfolio_sec if c.endswith("returns")], portfolio_name_dict=portfolio_name_dict)
    ax.set_title("Portfolio Performance on Test Set")
    fig.savefig(figure_path+"etf_sector_assets.png")
    plt.show()

    fig, ax = plot_portfolios(portfolio_bms,  ret_columns=[c for c in portfolio_bms if c.endswith("returns")], portfolio_name_dict=portfolio_name_dict)
    ax.set_title("Portfolio Performance on Train, Validation and Test Set")
    fig.savefig(figure_path+"etf_benchmark_all_phases.png")
    plt.show()


    fig, ax = plot_portfolios(portfolio_test_bms,  ret_columns=[c for c in portfolio_test_bms if c.endswith("returns")], portfolio_name_dict=portfolio_name_dict)
    ax.set_title("Portfolio Performance on Test Set")
    fig.savefig(figure_path+"etf_benchmark_test_phase.png")
    plt.show()

    fig, ax = plot_portfolios(portfolio_bms,  ret_columns=["BLEnd2End_returns"], portfolio_name_dict=portfolio_name_dict)
    ax.set_title("Portfolio Performance on Train, Validation and Test Set")
    fig.savefig(figure_path+"blend2end_all_phases.png")
    plt.show()

    # compare portfolios
    stats = compute_statistics_df(portfolio_all,  ret_columns=[c for c in portfolio_all if c.endswith("returns")], portfolio_name_dict=portfolio_name_dict)
    print(tabulate(stats, headers='keys', tablefmt='github', showindex=True, floatfmt=".3f"))
    with open(table_path+'stats.tex', 'w') as f:
        f.write(tabulate(stats, headers='keys', tablefmt='latex', showindex=True, floatfmt=".3f"))

    # compare sector only portfolios
    stats = compute_statistics_df(portfolio_sec,  ret_columns=[c for c in portfolio_sec if c.endswith("returns")], portfolio_name_dict=portfolio_name_dict)
    print(tabulate(stats, headers='keys', tablefmt='github', showindex=True, floatfmt=".3f"))
    with open(table_path+'sector_stats.tex', 'w') as f:
        f.write(tabulate(stats, headers='keys', tablefmt='latex', showindex=True, floatfmt=".3f"))

    # BLEnd2End compared to benchmark
    leverage = pd.read_pickle(bm_weights_path)
    stats = compute_statistics_df(portfolio_test_bms,  ret_columns=[c for c in portfolio_test_bms if c.endswith("returns")], portfolio_name_dict=portfolio_name_dict)
    stats.loc[ "BLEnd2End", "mean leverage"] = leverage.loc[leverage["phase"] == "test", "leverage"].mean()
    stats.loc["5-Factor Benchmark", "mean leverage"] = leverage.loc[leverage["phase"] == "test", "mvpo_leverage"].mean()
    print(tabulate(stats, headers='keys', tablefmt='github', showindex=True, floatfmt=".3f"))

    # Saving to .tex file
    with open(table_path+'benchmark_stats.tex', 'w') as f:
        f.write(tabulate(stats, headers='keys', tablefmt='latex', showindex=True, floatfmt=".3f"))

    # BLEnd2End in train, validation and test phase
    stats = pd.read_pickle(bl_performance_path)[["sharpe","sortino", "calmar", "mdd"]]
    leverage = pd.read_pickle(bm_weights_path)
    stats.loc["train", "mean leverage"] = leverage.loc[leverage["phase"] == "train", "leverage"].mean()
    stats.loc["val", "mean leverage"] = leverage.loc[leverage["phase"] == "val", "leverage"].mean()
    stats.loc["test", "mean leverage"] = leverage.loc[leverage["phase"] == "test", "leverage"].mean()

    ret_df = pd.read_pickle(bl_return_path)[["phase", "pred_ret_e_mean"]]
    for p in ["train", "val", "test"]:
        stats.loc[p, "mean return (%)"] = ret_df.loc[ret_df["phase"]==p,  "pred_ret_e_mean"].mean() * 100
        stats.loc[p, "vol (%)"] = ret_df.loc[ret_df["phase"]==p, "pred_ret_e_mean"].std() * 100

    stats = stats[["mean return (%)", "vol (%)", "sharpe","sortino", "calmar", "mdd"]]
    with open(table_path+'blend2end_train_val_test_stats.tex', 'w') as f:
        f.write(tabulate(stats, headers='keys', tablefmt='latex', showindex=True, floatfmt=".3f"))
    print(tabulate(stats, headers='keys', tablefmt='github', showindex=True, floatfmt=".3f"))

    # Benchmark performance in trian, validation and test phase
    stats = pd.read_pickle(bm_performance_path)[["sharpe","sortino", "calmar", "mdd"]]
    ret_df = pd.read_pickle(bm_weights_path)[["date_id", "phase", "mvpo_ret"]]
    ret_df = ret_df.groupby("date_id").last().reset_index(drop=True)
    stats.loc["train", "mean leverage"] = leverage.loc[leverage["phase"] == "train", "mvpo_leverage"].mean()
    stats.loc["val", "mean leverage"] = leverage.loc[leverage["phase"] == "val", "mvpo_leverage"].mean()
    stats.loc["test", "mean leverage"] = leverage.loc[leverage["phase"] == "test", "mvpo_leverage"].mean()

    for p in ["train", "val", "test"]:
        stats.loc[p, "mean return (%)"] = ret_df.loc[ret_df["phase"]==p, "mvpo_ret"].mean() * 100
        stats.loc[p, "vol (%)"] = ret_df.loc[ret_df["phase"]==p, "mvpo_ret"].std() * 100

    stats = stats[["mean return (%)", "vol (%)", "sharpe", "sortino", "calmar", "mdd"]]
    with open(table_path + 'benchmark_train_val_test_stats.tex', 'w') as f:
        f.write(tabulate(stats, headers='keys', tablefmt='latex', showindex=True, floatfmt=".3f"))
    print(tabulate(stats, headers='keys', tablefmt='github', showindex=True, floatfmt=".3f"))
