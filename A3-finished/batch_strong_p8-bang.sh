#!/bin/bash
# This script is intepreted by the Bourne Shell, sh
#
# Documentation for SGE is found in:
# http://docs.oracle.com/cd/E19279-01/820-3257-12/n1ge.html
#
# Tell SGE which shell to run the job script in rather than depending
# on SGE to try and figure it out.
#$ -S /bin/bash
#
# Export all my environment variables to the job
#$ -V
# Tun the job in the same directory from which you submitted it
#$ -cwd
#
# Give a name to the job
#$ -N nbody-strong-scale
#
#$ -pe orte 8
# Specify a time limit for the job
#$ -l h_rt=00:02:00
#
# Join stdout and stderr so they are reported in job output file
#$ -j y
#
# Run on the mpi queue, not more than 2 nodes
#$ -q mpi.q
#
# Specifies the circumstances under which mail is to be sent to the job owner
# defined by -M option. For example, options "bea" cause mail to be sent at the 
# begining, end, and at abort time (if it happens) of the job.
# Option "n" means no mail will be sent.
#$ -m aeb
#
# *** Change to the address you want the notification sent to
#$ -M yourLogin@ucsd.edu
#

# Change to the directory where the job was submitted from
cd $SGE_O_WORKDIR

echo
echo " *** Current working directory"
pwd
echo
echo " *** Compiler"
# Output which  compiler are we using and the environment
mpicc -v
echo
echo " *** Environment"
printenv

echo

echo ">>> Job Starts"
date

# Commands go here

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
date

#
echo ">>> Job Ends"
