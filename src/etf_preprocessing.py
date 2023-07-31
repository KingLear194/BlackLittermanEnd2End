import pandas as pd
import pickle, json, glob, itertools, functools, operator
#import torch
import settings, aux_utilities, train_utils

# Note: to have solvers everywhere, do solvers for both bl_updater and for weights calculation
project_name = 'etf'
load_bench_covs = f"{settings.trainpath}/{project_name}/"
'''
Global training parameters
'''
device = 'cuda'
nr_assets = 14
test_share = 0.3
risk_av = 5
trad_param = 0.005
train_val_split = 0.3 # size of val within train_val 
dropout = 0.05
early_stopping = (8,0)
leverage_type_constraint = 'L2'
batch_leverage_constraint = 0.5 

# comment about combinations within a job for etf: 2 x 3 x 2 x 4 x 4 x 3 = 576 jobs = 12 x 48

benchmark_pca_facs = (3,5) 
mults = (2,4,8) 
weight_solver_criteria = ['batch_foc_loss','batch_data_loss'] 

'''
ViewNet Parameters
'''
viewnet_types = ['FFNStandardTorch', 'RecurrentNNStandardTorch']
vnet_ffn_hidden_units = ((64,32,16), (64,64,32,32,16,16)) 
vnet_rnn_hidden_states = (4,8) 

viewnet_combos = [(viewnet_types[0], hidden_units) for hidden_units in vnet_ffn_hidden_units]\
    +[(viewnet_types[1], hidden_states) for hidden_states in vnet_rnn_hidden_states]
seq_len=6
num_layers = 1

'''
Solvers parameters
'''
lr_solvers = 0.001
nr_iterations_solvers = 40

'''
BL updater parameters
'''

bl_updater_types = ['bayesian_updater', 'KL','JSD', 'WSD2'] 
tau_views = 1/7 
shrink_fac_bl_cov=10

'''
Weight solver parameters
'''

weight_solver_types = [["weights_solver",weight_solver_criterion] for weight_solver_criterion in weight_solver_criteria]\
    +[["weights_net"], ['weightdiff_net']] 
hidden_units_static_weights_solver = [128, 64, 64, 32, 32]

def init_other_args():
    other_args = {}
    other_args['device'] = device
    other_args['nr_assets'] = nr_assets
    other_args['test_share'] = test_share
    other_args['risk_av'] = risk_av
    other_args['trad_param'] = trad_param
    other_args['batch_leverage_constraint'] = batch_leverage_constraint
    other_args['train_val_split'] = train_val_split
    other_args['dropout'] = dropout
    other_args['early_stopping'] = early_stopping
    other_args['seq_len'] = seq_len
    other_args['num_layers'] = num_layers
    other_args['lr_solvers'] = lr_solvers
    other_args['nr_iterations_solvers'] = nr_iterations_solvers
    other_args['tau_views'] = tau_views
    other_args['shrink_fac_bl_cov'] = shrink_fac_bl_cov
    other_args['hidden_units_static_weights_solver'] = hidden_units_static_weights_solver
    
    return other_args
    

def produce_algo_params(mult, viewnet_def, 
                        bl_updater_type=None,
                        weights_solver_type=None, 
                        other_args={}):
    
    if viewnet_def[0]=='RecurrentNNStandardTorch':
        viewnet_params = {'kind': 'GRU',
         'num_features':other_args['nr_view_features'],
         'seq_len':other_args['seq_len'],
         'output_size':other_args['nr_views'],
         'hidden_dim':viewnet_def[1],
         'num_layers':other_args['num_layers'],
         'output_only':True,
         'dropout':other_args['dropout'], 
         'device':other_args['device']}
        viewnet = (viewnet_def[0], viewnet_params)
    elif viewnet_def[0]=='FFNStandardTorch':        
        viewnet_params = {'hidden_units':viewnet_def[1],
                          'num_features':other_args['nr_view_features'],
                            'activation':'SiLU',
                            'bias':True,
                            'output_size':other_args['nr_views'],
                            'batchnormalization':True,
                            'dropout':other_args['dropout']}
        viewnet = (viewnet_def[0], viewnet_params)
    else:
        raise ValueError("Wrong name for viewnet!")
    
        
    if bl_updater_type=='bayesian_updater':
       bl_updater=('bayesian_updater', {'tau_view' : other_args['tau_views'],
                                        'n' : other_args['n'], 
                                        'device' : other_args['device']})
    elif bl_updater_type in ('JSD', 'WSD2', 'KL'):
       bl_updater = ('solver_updater',{'n' : other_args['n'],'K' : other_args['shrink_fac_bl_cov'],
                                       'nr_iterations':other_args['nr_iterations_solvers'],
                                       'optimizer_params':{'lr':other_args['lr_solvers']},
                                       'loss_name':bl_updater_type,
                                       'device' : other_args['device']})
    else:
        raise ValueError("Wrong name for BL updater!")
    
    if weights_solver_type[0]=='weights_solver':
        weights_calculator=(weights_solver_type[0], {'n':other_args['n'],'nr_iterations':other_args['nr_iterations_solvers'],
                                                   'optimizer_params':{'lr':other_args['lr_solvers']},
                                                   'risk_av':other_args['risk_av'],'tr_costs':other_args['trad_param'],
                                                   'loss_name':weights_solver_type[1],'device':other_args['device']})
    elif weights_solver_type[0] in ("weights_net", 'weightdiff_net'):
        weights_calculator=(weights_solver_type[0], {'n':other_args['n'],
                                                  'hidden_units':[other_args['hidden_units_static_weights_solver'] for _ in range(mult)],
                                                   'activation':'SiLU', 
                                                    'bias':True,
                                                    'dropout':other_args['dropout'], 
                                                    'risk_av':other_args['risk_av'],
                                                    'trad_param':other_args['trad_param'],
                                                    'device' :other_args['device']})
    else:
        raise ValueError("Wrong name for weights calculator!")

    return viewnet, bl_updater, weights_calculator


def produce_job(jobnr, benchmark_pca_fac, mult, 
                    viewnet_def, bl_updater_type, weights_calculator_type, dataset_type = 'l1norm'):
    
    '''
    viewnet, bl_updater, weights_calculator are pairs with (name, params)
    '''
    other_args=init_other_args()
    
    savepath = settings.trainpath+f"{project_name}/{jobnr}/"
    aux_utilities.create_dir_ifnexists(savepath)
    
    n = [other_args['nr_assets']]*mult
    other_args['n'] = n
    
    benchmark_folder = f'{project_name}_b_{benchmark_pca_fac}fac'
    
    # the initial offset in date_ids (i.e. number of zero rows we have to add to view_features for correct indexing purposes) has to equal the minimum date available for benchmarks. 
    # This is because the date_ids start from 1.  
    date_id_offset = min([int(file.replace(f"{settings.datapath}/final/{benchmark_folder}/","").replace("_cov.npy","").replace(f"{project_name}_b_","")) 
                          for file in glob.glob(f"{settings.datapath}/final/{benchmark_folder}/*_cov.npy")])#-1
    
    config_dict = {'batch_size':other_args['nr_assets']*mult,
                   'train_val_split':other_args['train_val_split'], 
                   'savepath':savepath,
                   'benchmark_folder':benchmark_folder,
                   'early_stopping':other_args['early_stopping'], 
                   'risk_av':other_args['risk_av'],
                   'trad_param':other_args['trad_param'], 
                   'optimizer_name':'Adam',
                   'train_criterion': 'batch_data_loss',
                   'test_criterion':'batch_data_loss',
                   'val_criterion':'sharpe_ratio',
                   'perf_measures':('sharpe_ratio', 'sortino_ratio', 'calmar_ratio'),
                   'load_benchcovs':load_bench_covs+f"{project_name}_b_{benchmark_pca_fac}fac_dict",
                   'device':other_args['device']}
    
    view_features = pd.read_pickle(f"{settings.finalpath}/{project_name}_macro_ts.pkl").sort_values('date')
    main_data = pd.read_pickle(f"{settings.finalpath}/{project_name}_{dataset_type}_dataset.pkl")#.sort_values(by = ['date_id','rix'])
    
    print("Throwing out COVID times")
    main_data = main_data[main_data['date']<='2020-01-01']
    
    view_weight_cols = [col for col in main_data.columns if col.endswith('weight')]
    other_args['nr_views']= len(view_weight_cols)
    
    print(f"Number of views is {other_args['nr_views']}")
    
    data, _ , _ = aux_utilities.enum_dates_assets(main_data, date_idx = 'date', asset_idx = 'asset_id',  
                                        save_path=f"{settings.main_path}/Data/current_train_val_test/{project_name}/{jobnr}", 
                                        job_number = jobnr)
    ixmap = data[['date_id', 'asset_id', 'asset_dt_pos']].sort_values(by = ['date_id','asset_id']).copy()
    enum_dates_assets = data[['rix','date_id','date','asset_id','asset_idx','asset_dt_pos']].copy()
    data = data[['rix','date_id','asset_id','ret_e', *view_weight_cols]].sort_values('rix')
    
    print("\n\ndate_id max now is ", data.date_id.max())
    
    view_features['date_id'] = view_features.index + 1
    view_features = view_features[view_features['date_id']>=date_id_offset].drop(columns = ['date'])
    other_args['nr_view_features'] = view_features.shape[1]-1
    print(f"The type of viewnet is {viewnet_def[0]}\n")
    print(f"Number of view features is {other_args['nr_view_features']}")
    
    data = data[data['date_id']>=date_id_offset]
    
    
    if viewnet_def[0]=='RecurrentNNStandardTorch':
        view_features = aux_utilities.df_lags_builder(view_features.set_index('date_id'), 5).reset_index().drop(columns = ['date_id'])
        date_id_offset+=other_args['seq_len'] # offset number for dates we lose due to RNN preprocessing and covariance estimation. The default value here is zero!
        view_features['date_id'] = range(date_id_offset,date_id_offset+view_features.shape[0])
    
    config_dict['date_id_offset']=date_id_offset
    print(f"\nfinal date_id_offset is {config_dict['date_id_offset']}")
    
    print(f"\nGetting ret_e_b from {settings.datapath}/final/{benchmark_folder}")
    
    
    ret_e_b, ret_e_b_dates = train_utils.get_bench_rets(project = project_name, path=f"{settings.datapath}/final/{benchmark_folder}", mindate=data.date_id.min(), maxdate = data.date_id.max())
    data['ret_e_b']=ret_e_b
    data = data[['rix','date_id','asset_id','ret_e','ret_e_b', *view_weight_cols]]
    
    data = data[data['date_id']>=config_dict['date_id_offset']].reset_index().drop(columns = ['index'])
    ixmap = ixmap[ixmap['date_id']>=config_dict['date_id_offset']].reset_index().drop(columns = ['index'])
    config_dict['ixmap_offset'] = int(data['rix'][0])
    
    print(f"\nixmap_offset is {config_dict['ixmap_offset']}")
    
    print("\nCheck for nans and infs: ixmap, data, view_features")
    aux_utilities.raise_naninf(ixmap)
    aux_utilities.raise_naninf(data)
    aux_utilities.raise_naninf(view_features)
    aux_utilities.raise_naninf(enum_dates_assets)
    
    valend = int(ixmap['date_id'].max() *(1-test_share))
    print("Valend is ", valend)
    X_test = data[data['date_id']>valend]#.drop(columns = ['date_id'])
    X_train_val = data[data['date_id']<=valend]#.drop(columns = ['date_id'])
    
    viewfeat_cols = [col for col in view_features.columns if col!='date_id']
    view_features = view_features[['date_id',*viewfeat_cols]]
    view_features.drop_duplicates().to_pickle(f"{savepath}/view_features.pkl")
    ixmap.to_pickle(f"{savepath}/ixmap.pkl")
    enum_dates_assets.to_excel(f"{savepath}/enum_dates_assets_{jobnr}.xlsx")
    X_train_val.to_pickle(f"{savepath}/X_train_val.pkl")
    X_test.to_pickle(f"{savepath}/X_test.pkl")
    
    print("\n\nDefining architectures and the other training parameters")
        
    viewnet, bl_updater, weights_calculator = produce_algo_params(mult, viewnet_def, bl_updater_type, weights_calculator_type, other_args)
    
    archidefs = {}
    archidefs[f'{jobnr}'] = {}
    archidefs[f'{jobnr}']['view_network_params']= viewnet
    archidefs[f'{jobnr}']['weight_calc_params'] = weights_calculator
    archidefs[f'{jobnr}']['bl_updater_params'] = bl_updater

    archi = open(f'{savepath}/archidefs.pkl', 'wb')
    pickle.dump(archidefs, archi)
    archi.close()
    with open(f"{savepath}/config_file.json", "w") as outfile:
        json.dump(config_dict, outfile, cls = aux_utilities.NpEncoder) 
    outfile.close()
    
    job_description = f"Job description for jobnumber {jobnr}\
        \n\nBenchmark PCA factors = {benchmark_pca_fac}\
        \n\nrisk_av = {other_args['risk_av']}, trad_param = {other_args['trad_param']}\
        \n\ntrain_test_criterion = 'batch_data_loss'\
        \n\nStatic sub-batch sizes = {other_args['n']}, total batch-size = {sum(other_args['n'])}\
        \n\nleverage type on the sub-batch level = {leverage_type_constraint}\
        \n\nleverage constraint on the sub-batch level = {other_args['batch_leverage_constraint']}\
        \n\ndropout = {other_args['dropout']}\
        \n\nearly_stopping = {other_args['early_stopping']}\
        \nArchitectures for BL Updater, ViewNet and Weight Calculation:\
        \n\nBLUpdater = {bl_updater[0]} with params: {bl_updater[1]}\
        \n\nViewNet_type={viewnet[0]} with params: {viewnet[1]}\
        \n\nWeights Calculator = {weights_calculator[0]} with params {weights_calculator[1]}"
    description = open(f"{savepath}/job_description_{jobnr}.txt", "w")
    description.write(job_description)
    description.close()
    
    print(f"\n\nFinished preprocessing for jobnumber {jobnr} with job description \n{job_description}")
    
def exec_preprocessing():

    print(f"Started Preprocessing for project {project_name}.\n\n")
    
    if load_bench_covs:
        print("Saving benchmark covariances as a dict for loading to computing units.")
        for nr in benchmark_pca_facs:
           train_utils.pickle_covs(cov_path = f"{settings.datapath}/final/{project_name}_b_{nr}fac/", 
                        savepath = f"{settings.datapath}current_train_val_test/{project_name}/{project_name}_b_{nr}fac_dict", project_name = project_name)
            
    hyperparams = (benchmark_pca_facs, mults, viewnet_combos, bl_updater_types, weight_solver_types)
    print(f"We have {functools.reduce(operator.mul, map(len, hyperparams),1)} hyperparameter combinations.\n")
    hyperparams = itertools.product(*hyperparams)
    
    jobnr=0
    
    for benchmark_pca_fac, mult, viewnet_def, bl_updater_type, weight_calculator_type in hyperparams:
    
        jobnr+=1
        print(f"\n\nPreprocessing for job {jobnr}:")
        produce_job(jobnr, benchmark_pca_fac, mult, 
                        viewnet_def, bl_updater_type, weight_calculator_type)
        
    
    print("Done with preprocessing for ETFs!!")

if __name__=='__main__':

    exec_preprocessing()
