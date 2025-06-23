import numpy as np
import scipy.interpolate as In
import time 
import sys 
import pickle 

from sys import argv
import config_PSO as cfg
from myPSO_cl import myPSO

from pyswarms.single.global_best import GlobalBestPSO
#from pyswarms.single.general_optimizer import GeneralOptimizerPSO
#from pyswarms.backend.topology import Pyramid
#from pyswarms.backend.topology import Random


if (__name__ == '__main__'):
	
	
	
	N = int(argv[1]) # n° seismometers
	NCluster = int(argv[2])		
	
	PSO = myPSO(cfg.xm, cfg.xM, cfg.ym, cfg.yM, cfg.InterPolIm, cfg.InterPolRe)
	xs = np.linspace(cfg.xm,cfg.xM, 151) #SIMS NEEDS A EVEN NUMBER OF INTERVALS; SO AN ODD NUMBER OF xs AND ys
	ys = np.linspace(cfg.ym,cfg.yM, 151)
	
	xx = -5.6
	yx = 0
	
	xy = -5.6
	yy = 0



	print('Starting optimization')
	t0 = time.time()
	
	# initiate the optimizer
	x_max = np.tile([cfg.xM - 0.01,cfg.yM - 0.01], N)
	x_min = np.tile([cfg.xm + 0.01,cfg.ym + 0.01], N)
	bounds = (x_min, x_max)
	options = {'c1': 0.5, 'c2': 0.9, 'w': 0.9}
	npart = 52
	niter = 10000
#	optimizer = GeneralOptimizerPSO(n_particles=npart, dimensions=2*N, options=options, bounds=bounds, topology=Pyramid(static=False))
	optimizer = GlobalBestPSO(n_particles=npart, dimensions=2*N, options=options, bounds=bounds)
	
	Final_State = optimizer.optimize(PSO.Res, niter, n_processes=8, xs=xs, ys=ys, xx=xx, yx=yx, xy=xy, yy=yy, dim=(len(ys), len(xs)), ns=N)
	cost_history = optimizer.cost_history
	Final_State = Final_State + (options, {'npart': npart, 'niter':niter}, cost_history)
	t1 = time.time()
	print('infinitely long elapsed time : ', t1-t0)

	filehandler = open('FS_N' + str(N) + '_' + str(NCluster) + '_h.obj', 'wb') 
	pickle.dump(Final_State, filehandler)
	filehandler.close()	
