#!/bin/bash
#
#SBATCH --job-name=etf_ensembling
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jed169@pitt.edu
#SBATCH --cluster=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --mem=48gb
#SBATCH --time=5-22:00:00
#SBATCH --chdir=/ix/jduraj/Bonds/Code
#SBATCH --output=/ix/jduraj/ResearchWithZihaoAndChen/Data/current_train_val_test/out_%x_%j.out

# change directory to the coding dir
cd /ix/jduraj/ResearchWithChen/Code/BLEndToEnd/src;

# activate environment  ---- change this as needed
source activate_env.sh;

now="$(date "+%d-%m-%H%M")";
SECONDS=0;

project_name='etf'; # ----- needs to match from preprocessing file
nrepochs=40;
jobnums=$(seq 1 576)

echo "##########################################################################################################################################"
echo "Project name is $project_name";

#echo "jobnumbers are $jobnums";

#hyperbest="$(python best_hyper.py -proj $project_name -val_crit sharpe_best_val -jobnr $jobnums)"
hyperbest="$(python best_hyper.py -proj $project_name -jobnr $jobnums)"
#echo "hyperbest is $hyperbest";
hyperb="$(echo "$hyperbest" | cut -d',' -f1)";
echo "Best hyperparams are $hyperb";
jobnum="$(echo "$hyperbest" | cut -d',' -f2 | tr -d ' ')"; #for further use down the road

randomstates=(19 10 12 26 100);
# ensemble predictions

echo "Now calculating the ensemble members for winning job number $jobnum with random states ${randomstates[@]}";
config_file_path="/ix/jduraj/ResearchWithZihaoAndChen/Data/current_train_val_test/$project_name/$jobnum/config_file.json";
for rstate in "${randomstates[@]}"
	do
		CUBLAS_WORKSPACE_CONFIG=:16:8 python one_shot_train.py -epochs $nrepochs $hyperb -rs $rstate -proj $project_name -config $config_file_path -time_start $now;
	done;

echo "Ensembling predictions";
python model_ensembling.py -projfolder $project_name $hyperb;

duration=$SECONDS;
duration_hrs=$(($duration/3600));
echo "Script took $duration_hrs hours, $((($duration-$duration_hrs*3600)/60)) minutes and $((($duration-$duration_hrs*3600)%60)) seconds to run";

conda deactivate;

echo "##########################################################################################################################################"

