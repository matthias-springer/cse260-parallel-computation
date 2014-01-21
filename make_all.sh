#!/bin/bash
for i in {1..8}; do
	for j in {1..8}; do
		#make VALIDATION=0 BSI=$[i*8] BSJ=$[i*8] BSK=$[j*8]
		#mv benchmark-blocked builds/blocked_$[i*8]_$[j*8]
		cat run_template > builds/blocked_$[i*8]_$[j*8].sh
		echo "builds/blocked_$[i*8]_$[j*8]" >> builds/blocked_$[i*8]_$[j*8].sh
		chmod +x builds/blocked_$[i*8]_$[j*8]
		chmod +x builds/blocked_$[i*8]_$[j*8].sh
	done
done

