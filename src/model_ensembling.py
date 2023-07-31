import settings
import argparse
import pandas as pd
import numpy as np
import performancemeasures as pm
import os


def produce_pathtail(ar):
    ar.path_tail = f"jnr_{ar.jobnumber}_bs_{ar.batchsize}_lr_{ar.learning_rate}_l2reg_{ar.l2regparam}_l1reg_{ar.l1regparam}_"


def main():

    arg_parser = argparse.ArgumentParser(description='Arguments for computing model performances and ensembling')
    arg_parser.add_argument('-projfolder', '--proj_folder', type=str, metavar='proj_folder')
    arg_parser.add_argument('-jobnr', '--jobnumber', type=int, default=75,
                            help='job number when submitting multiple jobs to the cluster')
    arg_parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, metavar='LR',
                            help='learning rate for global optimizer')
    arg_parser.add_argument('-l2reg', '--l2regparam', default='none', metavar='L2Reg',
                            help='weight for L2 regularization')
    arg_parser.add_argument('-l1reg', '--l1regparam', default='none', metavar='L1Reg_mom',
                            help='weight for L1 regularization')
    arg_parser.add_argument('-bs', '--batchsize', default=28, metavar='Batchsize', help='batch size')
    arg_parser.add_argument('-verbose', '--verbose', type=bool, default=True, metavar='verbose', help='verbosity (binary)')    
    args = arg_parser.parse_args()
    produce_pathtail(args)
    
    os.chdir(f"{settings.trainpath}{args.proj_folder}/{args.jobnumber}/")
   
    ret_df_paths = [filename for filename in os.listdir('.') 
                    if filename.startswith(f"pf_returns_{args.path_tail}rs") and filename.endswith('pkl')]
    weights_df_paths = [filename for filename in os.listdir('.') 
                    if filename.startswith(f"weights_{args.path_tail}rs") and filename.endswith('pkl')]
    
    assert(len(ret_df_paths)==len(weights_df_paths))
    if args.verbose: 
        print('Following files matches description')
        print(ret_df_paths, "\n\n")
        print(weights_df_paths)

    ret_dfs, weights_dfs = [], []
    for i in range(len(ret_df_paths)):
        ret_dfs.append(pd.read_pickle(ret_df_paths[i]))
        weights_dfs.append(pd.read_pickle(weights_df_paths[i]))

    if len(ret_dfs) != 0:
        # returns
        ens_ret_df = ret_dfs[0].drop(columns = ['pred_ret_e'])
        ret = np.concatenate([df['pred_ret_e'].to_numpy().reshape(-1,1) for df in ret_dfs], axis=1)
        ens_ret_df['pred_ret_e_mean'] = np.mean(ret, axis=1)
        ens_df = pm.produce_port_statistics(ens_ret_df, ret_label = 'pred_ret_e_mean')
        print(ens_df)
        ens_df.to_pickle(f"pf_performance_{args.path_tail}_mean_ensembled.pkl")
        ens_ret_df.to_pickle(f"pf_returns_{args.path_tail}_mean_ensembled.pkl")
        ens_df.to_excel(f"pf_performance_{args.path_tail}_mean_ensembled.xlsx")
        ens_ret_df.to_excel(f"pf_returns_{args.path_tail}_mean_ensembled.xlsx")
        
        ens_ret_df['drawdown'] = pm.get_drawdown_series(ens_ret_df, trainend = None, ret_label = 'pred_ret_e_mean')['drawdown']
        
        # weights 
        ens_weights_df = weights_dfs[0].drop(columns = ['weights'])
        weights = np.concatenate([df['weights'].to_numpy().reshape(-1,1) for df in weights_dfs], axis = 1)
        ens_weights_df['weights_mean'] = np.mean(weights, axis=1)
        ens_weights_df.to_pickle(f"weights_{args.path_tail}_mean_ensembled.pkl")
        ens_weights_df.to_excel(f"weights_{args.path_tail}_mean_ensembled.xlsx")        
        
        # aggregate to output
        xtrain = pd.read_pickle("X_train_val.pkl")[['rix','date_id','asset_id','ret_e']]
        xtest = pd.read_pickle("X_test.pkl")[['rix','date_id','asset_id','ret_e']]
        x = pd.concat([xtrain, xtest], axis = 0)
        ens_weights_df = x.merge(ens_weights_df, how = 'inner', on = 'rix').merge(ens_ret_df, how = 'inner', on = 'date_id')
        del xtrain, xtest
        
        if args.proj_folder.startswith('etf'):
            asset_idx = 'asset_idx'
        else:
            asset_idx = 'permno'
        
        if args.proj_folder.startswith('etf'):
            enum= pd.read_excel(f"enum_dates_assets_{args.jobnumber}.xlsx").drop(columns = ['Unnamed: 0','rix','asset_dt_pos']).drop_duplicates()
            ens_weights_df = enum.merge(ens_weights_df, how = 'inner', on = ['date_id','asset_id'])
        else:
            enum = pd.read_pickle([filename for filename in os.listdir('.') 
                            if filename.startswith("enum_dates_assets") and filename.endswith('pkl')][0])
            #ens_weights_df = x.merge(ens_weights_df, how = 'inner', on = 'rix').merge(ens_ret_df, how = 'inner', on = 'date_id')
            ens_weights_df = enum[[asset_idx,'asset_id']].drop_duplicates().merge(ens_weights_df, how = 'inner', on = 'asset_id')
            ens_weights_df = enum[['date_id','date']].drop_duplicates().merge(ens_weights_df, how = 'inner', on = 'date_id')
        ens_weights_df['leverage'] = ens_weights_df.groupby('date_id')['weights_mean'].transform(lambda x: x.abs().sum())
        ens_weights_df.drop(columns = ['date_id','asset_id','rix'], inplace = True)
            
        ens_weights_df = ens_weights_df[['date','phase',asset_idx,'ret_e','weights_mean','pred_ret_e_mean',
                                         'drawdown','leverage']]
        ens_weights_df.drop_duplicates().to_pickle(f"pf_output_{args.path_tail}_mean_ensembled.pkl")
        ens_weights_df.drop_duplicates().to_excel(f"pf_output_{args.path_tail}_mean_ensembled.xlsx")

        
    else:
        raise ValueError("No files in the directory matches described model!")


if __name__ == "__main__":
    main()