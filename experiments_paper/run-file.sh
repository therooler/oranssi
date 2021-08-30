#!/usr/bin/env bash
#SBATCH -c 16
#SBATCH -p cpu
#SBATCH --qos=nopreemption
#SBATCH --account=vector
#SBATCH --time=20:00:00
#SBATCH --mem 32G
#SBATCH --job-name=swap-swim
#SBATCH --output=outputs/slurm-%A_%a.out
SCRIPT=${1##}
export PATH=/pkgs/anaconda3/bin:$PATH
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate /h/roeland/condaenvs/pennylane/

echo $SCRIPT
python -u $SCRIPT
wait