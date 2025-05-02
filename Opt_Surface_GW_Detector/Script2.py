import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np

import time
from sys import argv
from scipy import signal

from pyKriging.krige import kriging  
from functools import partial
from multiprocessing import Pool

from itertools import combinations	
#%% LOAD
import pickle 

pickle_in = open("/storage/gpfs_small_files/VIRGO/users/fbadaracco/input/KL_scipy_mine.obj","rb")
KL = pickle.load(pickle_in)
pickle_in.close()

tmp = np.asarray(KL)

krigListRe = tmp[:,0]
krigListIm = tmp[:,1]

del tmp
lk = len(krigListIm)

nfft = 1024
fs = 250
Hann_win = signal.get_window('hann',nfft)
U = np.average(Hann_win**2, axis = 0)
fac = U*nfft*fs

def foo(klist):
	global lk
	global krigListIm
	global krigListRe
	global fac
	
	x1 = klist[0]
	y1 = klist[1]
	x2 = klist[2]
	y2 = klist[3]
	
	Fseg = np.zeros(lk, dtype = complex)
	for jj in range(0,lk):
		Fseg[jj] = (krigListRe[jj].predict([x1,y1]) - 1j*krigListIm[jj].predict([x1,y1]))*(krigListRe[jj].predict([x2,y2]) + 1j*krigListIm[jj].predict([x2,y2]))/fac	
	
	return np.average(Fseg)



#%% Sampling for 4D interpolation
if (__name__ == '__main__'):
	
	print('main')
	coords = np.array([[-2996.1855, 3.3109, -3.4011],[-2998.6876, 7.1706, -3.4071],[-2999.3596, 3.2421, -3.4008],[-3003.9149, 7.8757, -3.4227],[-3003.0811, 3.2749, -3.4150],[-3005.9728, 7.7190, -3.4288],[-3006.2434, 3.3279, -3.4237],[-3010.3192, 7.9888, -3.4437],[-3012.1375, 3.3488, -3.4299],[-3014.7164, 7.5541, -3.4490],[-3017.8701, 4.2944, -3.4613],[-3014.9770, -0.0874, -3.4426],[-3017.9104, -4.0994, -3.4595],[-3015.1017, -7.3702, -3.4573],[-3013.7171, -3.1819, -3.4287],[-3011.1778, -6.0311, -3.4400],[-3009.9611, -3.1498, -3.4302],[-3005.5196, -6.1313, -3.4285],[-3005.9568, -3.1540, -3.4234],[-3003.6825, -6.6868, -3.4195],[-3000.1694, -3.4201, -3.4149],[-2999.4935, -5.4172, -3.4146],[-2996.6250, -3.3377, -3.4085],[-2999.3105, 2.7366, -3.3156],[-3002.9757, 2.6436, -3.3247],[-3006.8761, 2.7944, -3.3304],[-3010.9353, 2.8204, -3.3416],[-3013.8210, 2.3477, -3.3491],[-3013.6882, -2.6371, -3.3435],[-3011.2803, -2.5983, -3.3446],[-3006.4946, -2.6779, -3.3386],[-3002.6733, -2.5689, -3.3296],[-2999.6753, -2.5736, -3.3223],[-2999.8896, 0.0321, -3.3182],[-3009.4183, 0.0789, -3.3497],[-3013.7110, -0.0613, -3.3462],[-3012.2423, -1.1159, -6.8445],[-3008.4083, 1.4434, -6.8571]])
	X = coords[:,0:2]
	X = np.delete(X,  [34], axis=0)

	x0 = -3005.5847
	y0 = 0.0312
	
	#centering around test mass
	X[:,0] = X[:,0] - x0
	X[:,1] = X[:,1] - y0
	
	xm = X[:,0].min()
	xM = X[:,0].max()
	ym = X[:,1].min()
	yM = X[:,1].max()
	
	Lx = xM - xm
	Ly = yM - ym
	
	
	"""
	We sample ns x ns points in the 2D space, then we divide this space in d x d squares (quarters), 
	each one identified by two coordinates: coord_qx and coord_qy and containing ns_q x ns_q points. 
	For each job we evaluate the CPSD between all the ns_q x ns_q points and all the points in the 2D space  (ns x ns).
	 
	argv[1] = number of points that we evaluate in the 2D space for create the thick grid (30)
	argv[2] = number of divisions made along each dimension (10)
	argv[3] = first coordinate of the quadrant (square)
	argv[4] = first coordinate of the quadrant (square)
	
	ns_q = number of points along one dimension in each division
	x1tot = x coordinate of the points in the quadrant with cordinates (coord_qx, coord_qy)
	y1tot = y coordinate of the points in the quadrant with cordinates (coord_qx, coord_qy)
	You then evaluate the CPSD between all the points in the quadrant with all the other points you created (30x30) and this is one single job. 
	You need to repeat the job for each quadrant.
	"""
	
	
	ns = int(argv[1])  #you put 14^2 seismometer on a regulare grid
	d = int(argv[2])
	print('placed: ', ns, ' seismometers with: ', d, ' divisions, -> ', ((ns//d)**2)*ns**2, ' evaluations in: ', d**2, ' jobs' )

	x1l = np.linspace(xm, xM, ns)
	y1l = np.linspace(ym, yM, ns)
	x2l = np.linspace(xm, xM, ns)
	y2l = np.linspace(ym, yM, ns)
	
	x1m,y1m,x2m,y2m  = np.meshgrid(x1l,y1l,x2l,y2l, indexing='ij')


	x1tot = x1m.flatten()
	y1tot = y1m.flatten()
	x2tot = x2m.flatten()
	y2tot = y2m.flatten()
	
	#selecting first quadrant
	ns_q = ns//d
	#coordinates quadrant
	coord_qx = int(argv[3])
	coord_qy = int(argv[4])
	
	indxtot = np.linspace(0,ns**4-1, ns**4, dtype = int)

	indx = np.array([])
	x1 = np.array([])
	y1 = np.array([])
	x2 = np.array([])
	y2 = np.array([])
	
	
	for sx in range(coord_qx*ns_q, coord_qx*ns_q + ns_q):
		for sy in range(coord_qy*ns_q, coord_qy*ns_q + ns_q):
			sel = (((x1tot == x1l[sx]) & (y1tot == y1l[sy])))	
			indx = np.append(indx,indxtot[sel])
			x1 = np.append(x1,x1tot[sel])
			y1 = np.append(y1,y1tot[sel])
			x2 = np.append(x2,x2tot[sel])
			y2 = np.append(y2,y2tot[sel])

	
	
	klistcoord = [None]*len(x1)
	for cc in range(0, len(x1)):
		klistcoord[cc] = [x1[cc], y1[cc], x2[cc], y2[cc]]

	lx1 = len(x1)

	print('starting sampling: len(x1, y1, x2, y2): ', len(x1), len(y1), len(x2), len(y2))

	t0 = time.time()	
	print('Starting pool ')
#	funx_p = partial(foo, lk) 
	Poo = Pool(8)
	Css = Poo.map(foo, klistcoord)
	print('Closing pool ')
	Poo.close()
	print('Joining pool ')
	Poo.join()

	t1 = time.time()
	print('Parallel: ', t1-t0)
	


	#%% SAVE
	import pickle 
	
	filehandler = open('Regular' + str(coord_qx) + str(coord_qy) + '_indx.obj', 'wb') 
	pickle.dump(indx, filehandler)
	filehandler.close()

	filehandler = open('Regular' + str(coord_qx) + str(coord_qy) + '_CSS.obj', 'wb') 
	pickle.dump(Css, filehandler)
	filehandler.close()
	
	filehandler = open('Regular' + str(coord_qx) + str(coord_qy) + '_x1.obj', 'wb') 
	pickle.dump(x1, filehandler)
	filehandler.close()
	
	filehandler = open('Regular' + str(coord_qx) + str(coord_qy) + '_y1.obj', 'wb') 
	pickle.dump(y1, filehandler)
	filehandler.close()
	
	filehandler = open('Regular' + str(coord_qx) + str(coord_qy) + '_x2.obj', 'wb') 
	pickle.dump(x2, filehandler)
	filehandler.close()
	
	filehandler = open('Regular' + str(coord_qx) + str(coord_qy) + '_y2.obj', 'wb') 
	pickle.dump(y2, filehandler)
	filehandler.close()
	




