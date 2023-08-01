# Black-Litterman End-To-End

*Jetlir Duraj, Chenyu Yu*

[Full paper](https://drive.google.com/file/d/12QNfdyKZ6XfImnju3Cl6GI-zStrKlruh/view?usp=sharing)

For a shorter project summary than the full paper go to ```summary.md``` (to be added soon). Until then, look at this interesting paper that covers the traditional model and a wide range of extensions:
[Kolm et al](https://www.pm-research.com/content/iijpormgmt/47/5/91). 

## Code organization 

The code for the project is under ```src```.
See the file ```requirements.txt``` for the packages under which code has been tested. Need to change paths as needed to run the scripts.

The file ```src/settings.py``` contains path information to fetch data and save results.

The preprocessing for the empirical exercise, creating the jobs submitted to the GPU cluster is in ```src/etf_preprocessing.py```.
The code to produce the benchmarks is in ```src/etf_noBL_performance.py```.

## How to run

To run an array of training jobs submit the script ```src/SlurmScripts/etf_training.sh```

    sbatch etf_training.sh

This loops over the jobs created by ```/src/etf_preprocessing.py```, $`L^1`$-regularization, $`L^2`$-regularization and learning rate for the global optimizer. 
The training for a fixed combination of job and hyperparameters uses ```/src/one_shot_train.py```

To fetch the validation results, find the best model, produce ensembling and final results, run the bash script ```src/SlurmScripts/etf_ensembling.sh```

    sbatch etf_ensembling.sh

