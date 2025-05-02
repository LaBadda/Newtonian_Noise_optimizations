# Underground Optimization

This folder contains three different scripts to optimize the seismic sensors around a single test mass in order to maximise Newtonian Noise cancellation performances (NB: no correlations between test masses are taken into account here).
These scripts assume seismic sensors with 1 measurement channel (i.e. only x instead of x,y,z).
The scripts where used to produce results for the following publication: [Optimization of seismometer arrays for the cancellation of Newtonian noise from seismic body waves](https://iopscience.iop.org/article/10.1088/1361-6382/ab28c1).

## How to use:

Hypotesis with only rayleigh field isotropic and homogeneus 

Test mass moving along x

To be run like:
	
	python3 nnOpt_DE_Ryleigh N hh
	
N = n° of seismometers
hh = Just a number that give the name to the Results<hh>.txt file in output

NOTE1: run multiple times this algorithms to find the best minimum (with hh from 1 to 100 or more...)
NOTE2: you can also decomment in the main :
```
        #pool = Pool(processes=6)
        #pool.map(foo, range(30))

```
and 
```
       #lock = Lock() 

```
inside the `foo()` function

and also comment:
```
        foo(int(argv[2]))
```
to run this in parallel on the local pc.


The frequency and other parameters relevant to the problem are built internally and must be changed in the script (this is ugly but I never had the time to modify it).
The script will produce a file called `Resultshh.txt` with hh equal to the number given in input. This file will contain all the relevant info for the optimization. This is not the best since then an external script will be needed to collect all the info for making plots and so on.   

  
