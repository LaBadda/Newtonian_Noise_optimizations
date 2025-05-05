# Optimizations scripts for Newtonian Noise cancellation in present and future GW detectors

Check the README.md in the subfolders. 


+ *Opt_Surface_GW_Detector*: contains the code used to optimize the seismic array in the gravitational wave detector Virgo using a data driven method.

+ *Opt_Underground_GW_Detector*: contains the code used to optimize the seismic array in an underground gravitational wave detector (the Einstein Telescope). It contains two subfolders: *Opt_single_TM* that runs the optimization around a single test mass (good in case of the end test mass of an L-shaped detector) and *Opt_4_TM* which assumes a triangular shape and the presence of correlated noise between the test masses at one corner. 
