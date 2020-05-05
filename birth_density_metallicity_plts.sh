#!/bin/bash -l
#SBATCH --ntasks 1 # The number of cores you need...
#SBATCH --cpus-per-task=8
#SBATCH -J FLARES-bdmet #Give it something meaningful.
#SBATCH -o logs/output_bdmet.%J.out
#SBATCH -e logs/error_bdmet.%J.err
#SBATCH -p cosma7 #or some other partition, e.g. cosma, cosma6, etc.
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 72:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=wjr21@sussex.ac.uk #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)

# Run the job from the following directory - change this to point to your own personal space on /lustre
cd /cosma7/data/dp004/dc-rope1/FLARES/flares

module purge
#load the modules used to build your program.
module load pythonconda3/4.5.4

source activate flares-env

# Run the program
#./metallicity_birth_density_plt_cumalative.py
./metallicity_birth_density_plt.py
#./metallicity_birth_density_plt_cumalative_REF.py
./metallicity_birth_density_plt_REF.py
./metallicity_birth_density_plt_meanden.py
./metallicity_birth_density_plt_AGNdT9.py
./metallicity_birth_density_plt_lowmass.py

source deactivate

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit

