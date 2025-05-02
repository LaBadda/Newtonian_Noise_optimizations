#!/bin/bash


cd /mnt/c/Users/franc/Documents/Projects/NNC_ET/OptimizationET/Risultati_3D_triangle_DE_Final_Nov_23/finalcode
for fo in $(ls .) 
do 
	cd $fo
	for N in $(ls .) 
	do 
		cd $N
		echo $fo/$N/
		[ -e En_N$N.txt ] && rm En_N$N.txt
		for file in $(ls Res*)
		do 
			cat $file|grep Energy|awk '{print $3 }' >> En_N$N.txt
		done
		cd ..
	done
	cd ..
done

