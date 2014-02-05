#!/bin/bash
for i in `seq 10 1199`;
do
	COMMAND[$i]=no_command
	BEST[$i]=0
done

for file in ./PROFILE_OUTUT_*
do
	first=1
	echo "processing ${file}..."
	while read line
	do
		assoc=( $line );
#		echo ${BEST[${assoc[0]}]}
#		echo "${assoc[1]} > ${BEST[${assoc[0]}]}"
		if [ $first -eq 0 ]; then
			if [ `echo "${assoc[1]} > ${BEST[${assoc[0]}]}" | bc -l` -eq 1 ]; then
				BEST[${assoc[0]}]=${assoc[1]}
				odd_increment=$(( ${assoc[0]} % 2 ))

				if [ "$file" == "./PROFILE_OLD" ]; then
					COMMAND[${assoc[0]}]="DGEMM_SELECT(${assoc[0]}, without_i, ${odd_increment}, 32, 32);"
#					COMMAND[${assoc[0]}]="transpose_B_${odd_increment}_${PARAMS_ARR[2]}_${PARAMS_ARR[2]}(lda, B); square_dgemm_without_i_${odd_increment}_32_32_32(lda, A, B_global_transpose, C);"
				else
					PARAMS=$(echo ${file} | tr "_" "\n")
					PARAMS_ARR=( ${PARAMS[0]} )
					COMMAND[${assoc[0]}]="DGEMM_SELECT(${assoc[0]}, with_i, ${odd_increment}, ${PARAMS_ARR[2]}, ${PARAMS_ARR[3]});"
#					COMMAND[${assoc[0]}]="transpose_B_${odd_increment}_${PARAMS_ARR[2]}_${PARAMS_ARR[2]}(lda, B); square_dgemm_with_i_${odd_increment}_${PARAMS_ARR[2]}_${PARAMS_ARR[2]}_${PARAMS_ARR[3]}(lda, A, B_global_transpose, C);"
				fi
			fi
		else
			# skip first line (meta data)
			first=0
		fi
	done < $file
done

for i in `seq 10 1199`
do
	#echo $i     ${BEST[$i]}
	echo "  ${COMMAND[$i]}"
done


