header = """!/bin/bash
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
#$ -N apf-real-b-weak
#
#$ -pe orte 64
# Specify a time limit for the job
#$ -l h_rt=00:10:00
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
#$ -M mspringe@ucsd.edu
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
"""

post = """
date
echo ">>> Job Ends"
"""
cnt = 0

for x in xrange(7):
	for y in xrange(7):
		#for M in [20, 40, 60, 80, 100, 120, 140, 160]:
		for M in [20]:
			if 2**x * 2**y > 64: continue
			f = open("run_weak_B_REAL_" + str(int(cnt /150)) + "_" + str(2**x) + "_" + str(2**y) + "_" + str(M) + ".sh", "w")
			cnt += 1
			f.write(header)
			for i in xrange(3):
				f.write("mpirun -np " + str((2**x) * (2**y)) + " ./nbody -s 1359856190 -M 100 -n " + str((((2**x) * (2**y)-1)/8+1) * 8192)  + " -px " + str(2**x) + " -py " + str(2**y) + "\n")
			f.write(post)
			f.close()
