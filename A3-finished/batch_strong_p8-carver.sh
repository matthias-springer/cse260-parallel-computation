#!/bin/csh
# This a batch submission script for a CUDA Job runing on Dirac
#  Submit this script using the command: qsub <script_name>
#  Use the "qstat" command to check the status of a job.
#
# There are only a few places you'll need to make changes, as marked below
 

# 1. The wall clock job time limit
# 2. The job name
# 3. An email notification when the job is complete
#   !!! Change to the address you want the notification sent to !!!
# 4. The job


#
# The following are embedded QSUB options. The syntax is #PBS (the # does
# _not_  denote that the lines are commented out so do not remove).
#

# Don't change this part of the batch submission file
#

#
# The job queues for Carver (we are not using Dirac)
# Up to 32 nodes (256 cores), 30 minutes, priority of 2 (lower than interactive)
#PBS -q debug
# Up to 64 nodes (512 cores), priority of 3 (lower than debug)
# PBS -q regular
#
# Ask for 1 node, all 8 CPU cores
#PBS -l nodes=1:ppn=8




# Job output files: you get separate files for stdout and stderr
#
# Filename for standard output (default = <job_name>.o<job_id>)
# at end of job, it is in directory from which qsub was executed

# Filename for standard error (default = <job_name>.e<job_id>)
# at end of job, it is in directory from which qsub was executed


# Export all my environment variables to the job
#PBS -V

#
# set echo               # echo commands before execution; use for debugging
#

# End of the part that shouldn't be changed

# ***  --- USer Changes start here ----
#
#
#
# 1. The requested wall clock job time limit in HH:MM:SS
#    Your job will end when it exceeds this time limit
#
#
#PBS -l walltime=00:05:00

#
# 2. *** Job name
#
# job name (default = name of script file)
# Change at will
#PBS -N NBODY-strong-8
#
#

#
# 3.  Your email address
#
#  ***********************
#  !!!!!!!!!!!!!!!!!!!!!!!!
#
# Change to the address you want the notification sent to!!
#PBS -M youremail@ucsd.edu
#
#  !!!!!!!!!!!!!!!!!!!!!!!!
#  ***********************
#
#
# Mail is sent when the job terminates normally or abnormally
# Add the letter 'b' to the end of the next line if you also want
# to be notified when the job begins to run
#PBS -m ae
#

#
# End of embedded QSUB options



# --------- The job starts ---------
# 4.  Put the executeable commands you want to run here
#	It is suggested that you always print out the environment
#	which maybe be useful in troubleshooting
printenv
date
echo "*** Job Starts"

#  You don't normally want to change this
cd $PBS_O_WORKDIR 	 #change to the working directory

#	It is suggested that you always print out the environment
#	which maybe be useful in troubleshooting
printenv
echo ">>> Job starts"

date

# The commands



# With the following settings you should be able to observe speed ups
# Sweep over all possible geometries

mpirun -np 1 ./nbody -t 75 -n 50000 -M 40 -s 1382741450 -px 1 -py 1
mpirun -np 2 ./nbody -t 75 -n 50000 -M 40 -s 1382741450 -px 1 -py 2
mpirun -np 2 ./nbody -t 75 -n 50000 -M 40 -s 1382741450 -px 2 -py 1
mpirun -np 4 ./nbody -t 75 -n 50000 -M 40 -s 1382741450 -px 1 -py 4
mpirun -np 4 ./nbody -t 75 -n 50000 -M 40 -s 1382741450 -px 2 -py 2
mpirun -np 4 ./nbody -t 75 -n 50000 -M 40 -s 1382741450 -px 4 -py 1
mpirun -np 8 ./nbody -t 75 -n 50000 -M 40 -s 1382741450 -px 1 -py 8
mpirun -np 8 ./nbody -t 75 -n 50000 -M 40 -s 1382741450 -px 2 -py 4
mpirun -np 8 ./nbody -t 75 -n 50000 -M 40 -s 1382741450 -px 4 -py 2
mpirun -np 8 ./nbody -t 75 -n 50000 -M 40 -s 1382741450 -px 8 -py 1



#

echo ">>> Job Ends"

#

cat /proc/cpuinfo

date

