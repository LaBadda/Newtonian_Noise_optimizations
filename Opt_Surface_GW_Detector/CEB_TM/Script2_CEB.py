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

pickle_in = open("KL_scipy_15Hz.obj","rb")
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
	X = np.zeros((49,2))

#without sensor numb 35 lev 2
	X[:,0] = np.array([ -1.8,  -7.8,  -7.8,  -1.8,  -1.8,   0.7,   0.7,   8.1,  -1.8, 0.7,   1.3,   6.8,   6.8,   3.9,   3.8,   1.5,   2.1,   2.1, -0.7,  -2.1,  -2.1,  -0.7,  -0.7,   0.6,  -5.7,  -8.5, -11.4, -7.7,  -6.3,  -3.5,  -2.1,  -3.4,   5.4,  -5.2,  -8.9, -10. , 5.4,   5.4,  -0.8,  -3.6,  -6.2,  -6.8, -12.2, -17.1, -19. , -18.6, -13.5,  14.8,  14.8])
	X[:,1] = np.array([ -1.8 ,  -1.8 ,   0.6 ,   1.  ,   7.4 ,   7.4 ,   0.8 ,   0.8 , -10.1 , -10.1 ,  -7.9 ,  -8.9 ,  -2.  ,  -2.4 ,  -5.3 ,  -4.9 , 6.65,   4.15,   3.65,   5.05,   7.25,   8.95,  11.85,   8.95, -2.  ,  -0.6 ,  -0.9 ,   1.25,   2.9 ,   0.9 ,  -0.1 ,  -1.4 , 9.2 ,   5.2 ,   4.8 ,   6.1 ,  13.3 ,  17.1 ,  17.9 ,  13.8 , -9.3 ,  -5.2 ,  -3.6 ,  -3.6 ,  -0.9 ,   3.7 ,   3.7 ,  -1.8 , 5.7 ])

	xm = X[:,0].min()
	xM = X[:,0].max()
	ym = X[:,1].min()
	yM = X[:,1].max()
	
	Lx = xM - xm
	Ly = yM - ym
	
	ns = int(argv[1])  
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
	




