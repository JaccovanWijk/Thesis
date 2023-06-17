#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH -t 24:00:00
#SBATCH --mail-type=BEGIN,END  
#SBATCH --mail-user=c.j.hourican@uva.nl
#SBATCH --array=0-62


module load 2020 
module load Anaconda3/2020.02

eval "$(conda shell.bash hook)"
conda activate synergy_env

python3 permutation_test_perTriplet.py $SLURM_ARRAY_TASK_ID



-----------------------------------------
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH -t 24:00:00
#SBATCH --mail-type=BEGIN,END  
#SBATCH --mail-user=jaccov.wijk@yahoo.com
#SBATCH --array=0-62

module load 2022

python -m pip install numpy scipy matplotlib pydotplus
python -m pip install pyagrum

python causalscore.py