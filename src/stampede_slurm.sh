
#!/bin/bash
#-----------------------------------------------------------------
# Example SLURM job script to run serial applications on TACC's
# Stampede system.
#
# This script requests one core (out of 16) on one node. The job
# will have access to all the memory in the node.  Note that this
# job will be charged as if all 16 cores were requested.
#-----------------------------------------------------------------

#SBATCH -J mysimplejob           # Job name
#SBATCH -o mysimplejob.%j.outfile    # Specify stdout output file (%j expands to jobId)
#SBATCH -p normal           # Queue name
#SBATCH -N 1                     # Total number of nodes requested (16 cores/node)
#SBATCH -n 16                     # Total number of tasks
#SBATCH -t 01:30:00              # Run time (hh:mm:ss) - 1.5 hours

#SBATCH -A DBS110003         # Specify allocation to charge against
#SBATCH --mail-user=katyasyc@gmail.com
#SBATCH --mail-type=ALL

# Load any necessary modules (these are examples)
# Loading modules in the script ensures a consistent environment.
module load hyperopt
module load tensorflow
module load numpy
module load scipy
module load scikit
module load python/2.7.3-epd-7.3.2
module load sklearn
module load cPickle
module load Queue
module load json
#etc.

# Launch the executable named "a.out"
./stampede_job.sh sst2 rand reg 5
./stampede_job.sh sst2 grid arch 5


""" high level on stampede:
dir for ceil(iters/16)
script to run 16
save to dir as 0_0-0_15
then next file 1_0-1_15
etc.
eval all dirs in save_dir

todo:
stampede codec
technical writeup
where to mkdirs?
why 6 procs instead of 4?? 3 "3" and 3 "2"
  I'm limiting the same way, but it's not limiting, as if paralellized to 3 ???
  Not "adding" to CPU usage of highest processed
  but not saving model of its own
  im stumped :(
  didn't happen when I run manually, I don't think

done:
eval.py (for the moment)
w2v processing
added wait to bash script :)"""
