#!/bin/bash
#
#SBATCH --job-name=etf_training
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jed169@pitt.edu
#SBATCH --cluster=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --mem=80gb
#SBATCH --time=5-23:00:00
#SBATCH --chdir=/ix/jduraj/Bonds/Code
#SBATCH --output=/ix/jduraj/ResearchWithZihaoAndChen/Data/current_train_val_test/out_%x_%j.out

# change directory to the coding dir
cd /ix/jduraj/ResearchWithChen/Code/BLEndToEnd/src;
source activate_env.sh;

now="$(date "+%d-%m-%H%M")";
SECONDS=0;
project_name='etf'; # ----- needs to match from preprocessing file
nrepochs=40;

# no batch size and architectures, because that needs to be fixed from the start
jobnums=( $(seq 1 576) ) 
learningrates=(0.001 0.0001); 
l1regs=("none" 0.01 0.1); 
l2regs=("none" 0.01 0.1); 
random_seeds=(0);

echo "##########################################################################################################################################"
echo "Project name is $project_name";
for jobnm in "${jobnums[@]}"
	do	
		echo "current jobnumber is $jobnm";
		config_file_path="/ix/jduraj/ResearchWithZihaoAndChen/Data/current_train_val_test/$project_name/$jobnm/config_file.json"; 
		for lrate in "${learningrates[@]}"
			do
				for l1reg in "${l1regs[@]}"
				  	do
						for l2reg in "${l2regs[@]}"
					   		do
		 				 		for rstate in "${random_seeds[@]}"
					        		do
			         CUBLAS_WORKSPACE_CONFIG=:16:8 python one_shot_train.py -epochs $nrepochs -lr $lrate -l1reg $l1reg -l2reg $l2reg -proj $project_name -jobnr $jobnm -rs $rstate -config $config_file_path -time_start $now;
									done
							done	
					done
			done
	done;

duration=$SECONDS;
duration_hrs=$(($duration/3600));
echo "Script took $duration_hrs hours, $((($duration-$duration_hrs*3600)/60)) minutes and $((($duration-$duration_hrs*3600)%60)) seconds to run";

conda deactivate;

echo "##########################################################################################################################################"

