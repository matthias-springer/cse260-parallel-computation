#!/bin/bash
for file in builds/*.sh
do
	qsub $file
done
