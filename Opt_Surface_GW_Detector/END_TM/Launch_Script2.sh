#!/bin/bash

d=10
m=3


for i in $(seq 0 $((d-1)))
do
	for j in $(seq 0 $((d-1)))
	do
		sed "s/echo \"Quadrants 0 0\"/echo \"Quadrants $i $j\"/" /storage/gpfs_small_files/VIRGO/users/fbadaracco/input/Template_Run_Script2.sh > /storage/gpfs_small_files/VIRGO/users/fbadaracco/RegGrid_scripts10/R_Reg${i}${j}.sh
		sed -i "s/KreeBirth_Reg_Local10.py 0 0 0 0/KreeBirth_Reg_Local10.py $(($d*$m)) $d $i $j/" /storage/gpfs_small_files/VIRGO/users/fbadaracco/RegGrid_scripts10/R_Reg${i}${j}.sh
		chmod 755 /storage/gpfs_small_files/VIRGO/users/fbadaracco/RegGrid_scripts10/R_Reg${i}${j}.sh

		echo $i $j
		bsub -q virgo -J Reg10 -n 8 -R "select [(mcore==1)] span[ptile=8]" -e stderr.txt -f "stderr10_${i}${j}.txt < stderr.txt" -o stdout.txt -f "stdout10_${i}${j}.txt < stdout.txt" /storage/gpfs_small_files/VIRGO/users/fbadaracco/RegGrid_scripts10/R_Reg${i}${j}.sh


	done
done


