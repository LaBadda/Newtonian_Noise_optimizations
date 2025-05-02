#!/bin/bash

cd /mnt/c/Users/franc/Documenti/Projects/NNC_ET/OptimizationET/Ultimi
for fo in $(ls .)
do
        cd $fo
        for N in $(ls .)
        do
                cd $N
                echo $fo/$N/
		cp /mnt/c/Users/franc/Documents/Projects/NNC_ET/OptimizationET/Script_da_tenere/template.py .
                printf "N = ${PWD##*/}\nE_vec = np.array(["  >>template.py
		for ii in $(ls Re*); do grep Energy $ii|awk '{printf("%f,", $3)}'>>template.py;  done
		printf "])\n">>template.py
		for ii in $(ls Re*); do grep Energy $ii>>template.py; grep Final  $ii| sed "s/c='g'/c = Energy*np.ones(N), vmin = min(E_vec), vmax = max(E_vec)/g" >> template.py;  done
		sed -i '1,/ax.scatter(Fin/s/ax.scatter(Fin/CC = ax.scatter(Fin/' template.py 
		printf "plt.colorbar(CC)\n" >> template.py
		printf "plt.savefig(\'/mnt/c/Users/franc/Documenti/Projects/NNC_ET/OptimizationET/Ultimi_Plot/Plot_N${N}_p$fo.png\', bbox_inches='tight')" >> template.py
		python3 template.py
		cd ..
        done
        cd ..
done
