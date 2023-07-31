import pandas as pd
import warnings
warnings.filterwarnings("ignore")


import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 6]
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

mpl.rc('font', **font)
mpl.style.use('seaborn-whitegrid')

pd.options.display.max_columns = None

main_path = "/ix/jduraj/ResearchWithChen"
#path = os.path.expanduser(f"{main_path}/")
datapath = f"{main_path}/Data/"
codepath = f"{main_path}/Code/src"
trainpath = f"{main_path}/Data/current_train_val_test/"
cov_path = f"{datapath}/final/etf_cov"

