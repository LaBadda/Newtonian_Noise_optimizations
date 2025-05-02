import numpy as np
import scipy.interpolate as In
import time 
import sys 
import pickle 

from sys import argv

from itertools import combinations	
from scipy.integrate import simps
from pyswarms.single.global_best import GlobalBestPSO
#from pyswarms.single.general_optimizer import GeneralOptimizerPSO
#from pyswarms.backend.topology import Pyramid
#from pyswarms.backend.topology import Random

from scipy import linalg

coords = np.array([[-2996.1855, 3.3109, -3.4011],[-2998.6876, 7.1706, -3.4071],[-2999.3596, 3.2421, -3.4008],[-3003.9149, 7.8757, -3.4227],[-3003.0811, 3.2749, -3.4150],[-3005.9728, 7.7190, -3.4288],[-3006.2434, 3.3279, -3.4237],[-3010.3192, 7.9888, -3.4437],[-3012.1375, 3.3488, -3.4299],[-3014.7164, 7.5541, -3.4490],[-3017.8701, 4.2944, -3.4613],[-3014.9770, -0.0874, -3.4426],[-3017.9104, -4.0994, -3.4595],[-3015.1017, -7.3702, -3.4573],[-3013.7171, -3.1819, -3.4287],[-3011.1778, -6.0311, -3.4400],[-3009.9611, -3.1498, -3.4302],[-3005.5196, -6.1313, -3.4285],[-3005.9568, -3.1540, -3.4234],[-3003.6825, -6.6868, -3.4195],[-3000.1694, -3.4201, -3.4149],[-2999.4935, -5.4172, -3.4146],[-2996.6250, -3.3377, -3.4085],[-2999.3105, 2.7366, -3.3156],[-3002.9757, 2.6436, -3.3247],[-3006.8761, 2.7944, -3.3304],[-3010.9353, 2.8204, -3.3416],[-3013.8210, 2.3477, -3.3491],[-3013.6882, -2.6371, -3.3435],[-3011.2803, -2.5983, -3.3446],[-3006.4946, -2.6779, -3.3386],[-3002.6733, -2.5689, -3.3296],[-2999.6753, -2.5736, -3.3223],[-2999.8896, 0.0321, -3.3182],[-3009.4183, 0.0789, -3.3497],[-3013.7110, -0.0613, -3.3462],[-3012.2423, -1.1159, -6.8445],[-3008.4083, 1.4434, -6.8571]])	
X = coords[:,0:2]	
X = np.delete(X,  [34], axis=0)#seismometer on metal platform

x0 = -3005.5847
y0 = 0.0312
#h = 1.5#np.average(coords[0:21,2]) - (-2.2233)

#centering around test mass
X[:,0] = X[:,0] - x0
X[:,1] = X[:,1] - y0

xm = X[:,0].min()
xM = X[:,0].max()
ym = X[:,1].min()
yM = X[:,1].max()

Lx = xM - xm
Ly = yM - ym


pickle_in = open('C:/Users/Francesca/Desktop/Results_and_kriging/Reg_out20/CSS_30x.obj',"rb")
CSS = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('C:/Users/Francesca/Desktop/Results_and_kriging/Reg_out20/x1_30x.obj',"rb")
x1 = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('C:/Users/Francesca/Desktop/Results_and_kriging/Reg_out20/y1_30x.obj',"rb")
y1 = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('C:/Users/Francesca/Desktop/Results_and_kriging/Reg_out20/x2_30x.obj',"rb")
x2 = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('C:/Users/Francesca/Desktop/Results_and_kriging/Reg_out20/y2_30x.obj',"rb")
y2 = pickle.load(pickle_in)
pickle_in.close()

Xcomb = np.stack((x1,y1,x2,y2), axis = 1)
CssIm = np.imag(CSS)
CssRe = np.real(CSS)

X = np.unique(x1)
Y = np.unique(y1)
Xm1, Ym1, Xm2, Ym2 = np.meshgrid(X,Y,X,Y, indexing='ij')
CSSregridRe = CssRe.reshape(Xm1.shape)
CSSregridIm = CssIm.reshape(Xm1.shape)

InterPolRe = In.RegularGridInterpolator((X,Y,X,Y), CSSregridRe, method='linear')
InterPolIm = In.RegularGridInterpolator((X,Y,X,Y), CSSregridIm, method='linear')
#InterPolRe = In.NearestNDInterpolator(Xcomb, CssRe)
#InterPolIm = In.NearestNDInterpolator(Xcomb, CssIm)

print('Interpolation done.')
def Css_matrix(Ip,ns):
	
	"""
	Ip are the positions of the ns seismometers coming from the optimization process
	
	"""
	Css_vecIm = InterPolIm.__call__(Ip)
	Css_vecRe = InterPolRe.__call__(Ip)

	Css = Css_vecRe.reshape(ns,ns) + 1j*Css_vecIm.reshape(ns,ns)
	
	if (np.any(np.isnan(Css))):
		print('Warning. NaN in Css : ', Css , file = sys.stderr)	

	Diag = np.diag(Css)
	Nfact = np.sqrt(np.tensordot(Diag,Diag, axes = 0))
	"""
	Nfact is a matrix where each elements is Nfact_ij = ASD_i ASD_j, with ASD the amplitude spectral density of the sensor i or j
	In the end Css become a coherence matrix 
	"""	
	Css = Css/Nfact
	
	"""
    why the svd: 
    with the singular value decomposition, you can write any matrix M as M = UEV and since U and V are 
    unitary, the inverse of M is easy to be calculated, iM is: iM = hU iE hV, where hU and hV are the 
    transposed conjugate of U and V, respectively. iE is easy to calculate since E is diagonal.
    If for some numerical problems you cannot calculate the inverse, this means that E must have some 
    diagonal value close to zero. You can remove these values from E and cut U and V accordingly to 
    reconstruct the pseudo-inverse of M. So, it is a trick to avoid problems in the inversion. 
    That function returns already the inverse of M.
    """
    
	[U,diagS,V] = linalg.svd(Css)
	#		S = np.diag(diagS)
	thresh = 0.01         
	kVal = np.sum(diagS > thresh)
	#		Css_svd = (U[:, 0: kVal]@np.diag(diagS[0: kVal])@V[0: kVal, :])#inverse of the reconstructed Css
	iU = (U.conjugate()).transpose()
	iV = (V.conjugate()).transpose()
	Css_svd = (iV[:,0: kVal])@np.diag(1/diagS[0: kVal])@(iU[0: kVal,:])#inverse of the reconstructed Css
	Css_svd = Css_svd/Nfact
	
	return Css_svd		
	
def Csn_vector(x,y, xs, ys, dim): 

	"""xs and ys are the points defining the grid over which perform the integral"""

	ns = len(x)
	Csn = np.zeros(ns, dtype = complex)
	for uu in range(0,ns):
		xsm, ysm = np.meshgrid(xs,ys)

		Ip = np.stack((xsm.flatten(), ysm.flatten(),x[uu]*np.ones(xsm.shape[0]*xsm.shape[1]), y[uu]*np.ones(xsm.shape[0]*xsm.shape[1])), axis = 1)
		

		Csn_v = InterPolRe.__call__(Ip) + 1j*InterPolIm.__call__(Ip)
		if ((x[uu]>-8)&(x[uu]<6) & (y[uu]>-2.6)&(y[uu]<2.6)):
			h = 5
		else:
			h = 1.5
		
		Csn_s = np.reshape(Csn_v,dim)*((xsm)/((h**2 + (xsm)**2 + (ysm)**2)**(3/2)))

		#	Integration along xs:
		Csnx = simps(Csn_s, xs, axis = 1)
		Csn[uu] = simps(Csnx, ys)

	if (np.any(np.isnan(Csn))):
		print('Warning. NaN in Csn: ', Csn , file = sys.stderr)	
	
	return Csn

def Res(state, xs, ys, dim, ns):

	#print(state.shape)
	n_part = state.shape[0]
	Res_Vec = np.zeros(n_part)
	
	for tt in range(0,n_part):
		s = state[tt,:].reshape(ns,2)
		x = s[:,0]
		y = s[:,1]
	
		# Css:
		# Generate the combinations relative to the matrix elements of Css:
		IDmat1 = np.zeros((ns, ns), dtype = int)
		IDmat2 = np.zeros((ns, ns), dtype = int)
		for ii in range(0,ns):
			for jj in range(0,ns):
				IDmat1[ii,jj] = ii
				IDmat2[ii,jj] = jj

		ID1 = IDmat1.flatten()
		ID2 = IDmat2.flatten()
		Ip = np.stack((x[ID1], y[ID1], x[ID2], y[ID2]), axis = 1)

		if (((min(x) < xm) | (max(x) > xM) | (min(y) < ym) | (max(y) > yM)) == 1):
			raise ValueError('Boundaries violated')

#		snr = 1e-20
		Css = Css_matrix(Ip,ns)# + np.diag(np.ones(ns)*snr)

		# Csn:
		Csn = Csn_vector(x,y, xs, ys, dim)
		if((np.sum(np.isnan(Csn))) & (np.sum(np.isnan(Css.flatten()))) >= 1):
			print("c'è un NaN(o)")
		

#		residual = 1 - np.dot(Csn.conjugate(),np.dot(Css,Csn))/(3.008987631861667e-13)
		residual = -np.dot(Csn.conjugate(),np.dot(Css,Csn))
		Res_Vec[tt] = np.real(residual)	
	
	return Res_Vec

	
if (__name__ == '__main__'):
	
	N = int(argv[1]) # n° seismometers
	NCluster = int(argv[2])		
	
#	dd = 0
#
#	#Uneven spacing along x and y
#	xpl1_0 = -9 - dd
#	xpl1_1 = -7 + dd
#	
#	xpl2_0 = 6 - dd
#	xpl2_1 = 8 + dd
#	
#	ypl1_0 = -4 - dd 
#	ypl1_1 = -2 + dd
#	
#	ypl2_0 = 2 - dd
#	ypl2_1 = 4 + dd
#	
#	d = 0.4
#	d2 = 1 
#	xs0 = np.arange(xm, xM, d)
#	ys0 = np.arange(ym, yM, d)
#	
#	
#	indx10 = min(np.argwhere((xs0 > xpl1_0)))[0]
#	indy10 = min(np.argwhere((ys0 > ypl1_0)))[0]
#	
#	indx11 = min(np.argwhere((xs0 > xpl1_1)))[0]
#	indy11 = min(np.argwhere((ys0 > ypl1_1)))[0]
#	
#	
#	
#	indx20 = min(np.argwhere((xs0 > xpl2_0)))[0]
#	indy20 = min(np.argwhere((ys0 > ypl2_0)))[0]
#	
#	indx21 = min(np.argwhere((xs0 > xpl2_1)))[0]
#	indy21 = min(np.argwhere((ys0 > ypl2_1)))[0]
#	
#	xs = np.append(xs0, np.arange(xs0[indx10], xs0[indx11],(xs0[indx10+1]-xs0[indx10])/d2))
#	ys = np.append(ys0, np.arange(ys0[indy10], ys0[indy11],(ys0[indy10+1]-ys0[indy10])/d2))
#	
#	xs = np.append(xs, np.arange(xs0[indx20], xs0[indx21],(xs0[indx20+1]-xs0[indx20])/d2))
#	ys = np.append(ys, np.arange(ys0[indy20], ys0[indy21],(ys0[indy20+1]-ys0[indy20])/d2))


	xs = np.linspace(xm,xM, 151) #SIMS NEEDS A EVEN NUMBER OF INTERVALS; SO AN ODD NUMBER OF xs AND ys
	ys = np.linspace(ym,yM, 151)
	
	print('Starting optimization')
	t0 = time.time()
	
	# initiate the optimizer
	x_max = np.tile([xM - 0.01,yM - 0.01], N)
	x_min = np.tile([xm + 0.01,ym + 0.01], N)
	bounds = (x_min, x_max)
	options = {'c1': 0.5, 'c2': 0.9, 'w': 0.9}
	npart = 52
	niter = 100#101000
#	optimizer = GeneralOptimizerPSO(n_particles=npart, dimensions=2*N, options=options, bounds=bounds, topology=Pyramid(static=False))
	optimizer = GlobalBestPSO(n_particles=npart, dimensions=2*N, options=options, bounds=bounds)
	
	# now run the optimization, pass a=1 and b=100 as a tuple assigned to args
	Final_State = optimizer.optimize(Res, niter, n_processes=4, xs=xs, ys=ys, dim=(len(ys), len(xs)), ns=N)
	cost_history=optimizer.cost_history
	Final_State = Final_State + (options, {'npart': npart, 'niter':niter}, cost_history)
	t1 = time.time()
	print('infinitely long elapsed time : ', t1-t0)

	filehandler = open('FS_N' + str(N) + '_' + str(NCluster) + '_h.obj', 'wb') 
	pickle.dump(Final_State, filehandler)
	filehandler.close()	
