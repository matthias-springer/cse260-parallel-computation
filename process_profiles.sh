#!/bin/bash
for i in `seq 10 1199`;
do
	BEST[$i]=0
#	echo ${BEST[$i]}
done

for file in ./PROFILE_OUTUT_*
do
	echo "processing ${file}..."
	while read line
	do
		assoc=( $line );
#		echo ${BEST[${assoc[0]}]}
#		echo "${assoc[1]} > ${BEST[${assoc[0]}]}"
		if [ `echo "${assoc[1]} > ${BEST[${assoc[0]}]}" | bc -l` -eq 1 ]; then
			BEST[${assoc[0]}]=${assoc[1]}
		fi
#		BEST[${assoc[0]}]=$((${assoc[1]}>$BEST[${assoc[0]}]?${assoc[1]}:$BEST[${assoc[0]}]))
#		BEST[${assoc[0]}]=${assoc[1]}
#		echo "${assoc[1]}"
	done < $file
done

for i in `seq 10 1199`
do
	echo $i     ${BEST[$i]}
done


