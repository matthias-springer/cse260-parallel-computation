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
#$ -N PROFILES
#
# Specify a time limit for the job
#$ -l h_rt=00:20:00
#
# Join stdout and stderr so they are reported in job output file
#$ -j y
#
# Run on the debug queue; only one node may be used
# To use more than one node, specify the "normal" queue
#$ -q debug.q
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

builds/blocked_24_56
