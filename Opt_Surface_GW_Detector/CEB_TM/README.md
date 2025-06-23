# Surface GW Detector - Data-driven optimization

This folder contains the scripts to run the seismic sensor optimization that were used in the following publication: [Environmental noises in current and future gravitational-wave detectors](https://iopscience.iop.org/article/10.1088/1742-6596/2156/1/012077). It considers correlations in the optimization of a seismic array for Newtonian noise cancellation in the central part of a Michelson interferometer, where correlations between the inner test masses must be considered.
We consider: 

Lx = L0 + dx(L0) - dx(0)
Ly = L0 + dy(L0) - dy(0)

where dy(L0) is the displacement caused at the end test mass and the dy(0) at the input one for y (and same for x).

So: 
Delta\_L = (dy(L0) - dx(L0)) - (dy(0) - dx(0))

But being that the only correlated terms are dy(0) and dx(0), the PSD becomes:

PSD(Delta\_L) = PSD(dy(L0)) + PSD(dx(L0)) + PSD( dy(0) - dx(0) )

So my target signal is: dy(0) - dx(0)

# How to use: 


Run `Script1_CEB.py`, then `Script2_CEB.py` (how to run this particular script is showen in `Script2_CEB.sub`). Then `Script3_CEB.py` and finally `Script4_CEB.py` which is the actual optimization. `Script1_CEB.py`, `Script2_CEB.py` and `Script3_CEB.py` all generate the Fourier transform and make the Gaussian Process Regression to estimate correlations. 
