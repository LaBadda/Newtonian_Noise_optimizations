# Underground Optimization

This folder contains the scripts to optimize the seismic sensors around a corner of a GW detector (having 4 test masses and having triangular shape).
Correlations between test masses are taken into account here.
These scripts assume seismic sensors with 3 measurement channel (i.e. x,y,z).
The scripts where used to produce results for the following publication: [Joint optimization of seismometer arrays for the cancellation of Newtonian noise from seismic body waves in the Einstein Telescope](https://iopscience.iop.org/article/10.1088/1361-6382/ad1715) 


## How to use:

Limit for cavity negligible respect to wavelength of the seismic field (ka->0). 
Seismometers with 3 measuring channels: x,y,z.
Test mass moving along x

To be run like:
```	
	python3 nnOpt_DE_bulk N hh workers
```
	
N = n° of seismometers
hh = Just a number that give the name to the Results<hh>.txt file in output
workers = workers parameter of the Particle Swarm function (`n_processes` parameter, to parallelize the optimization)

NOTE1: run multiple times this algorithms to find the best minimum (with hh from 1 to 100 or more...)


The frequency and other parameters relevant to the problem are built internally and must be changed in the script (this is ugly but I never had the time to modify it).
The script will produce a file called `Resultshh.txt` with hh equal to the number given in input. This file will contain all the relevant info for the optimization. This is not the best since then an external script will be needed to collect all the info for making plots and so on.   

  
