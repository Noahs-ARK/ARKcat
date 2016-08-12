#!/bin/bash
#-----------------------------------------------------------------
# Example SLURM job script to run serial applications on TACC's
# Stampede system.
#
# This script requests one core (out of 16) on one node. The job
# will have access to all the memory in the node.  Note that this
# job will be charged as if all 16 cores were requested.
#-----------------------------------------------------------------

#SBATCH -J test_conda           # Job name
#SBATCH -o test_conda.%j.outfile    # Specify stdout output file (%j expands to jobId)
#SBATCH -p normal           # Queue name
#SBATCH -n 5                     # Total number of tasks
#SBATCH -t 0:30:00              # Run time (hh:mm:ss) - 1.5 hours

#SBATCH -A TG-DBS110003        # Specify allocation to charge against
#SBATCH --mail-user=katyasyc@gmail.com
#SBATCH --mail-type=ALL

# Load any necessary modules (these are examples)
# Loading modules in the script ensures a consistent environment.
module load python/2.7.3-epd-7.3.2

./gaidheal.sh
