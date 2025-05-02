from scipy import special as sp
from scipy.optimize import basinhopping
import numpy as np
from scipy import linalg

import json
from time import clock
from sys import argv

'''

Limit for cavity negligible respect to wavelength of the seismic field (ka->0). 
Seismometers with 1 measuring channel.
Test mass moving along x

To be run like:
	
	python3 nnOpt_BH_bulk N hh
	
N = n° of seismometers
hh = Just a number that give the name to the Results<hh>.txt file in output

NOTE1: run multiple times this algorithms to find the best minimum (with hh from 1 to 100 or more...)
NOTE2: you can also decomment in the main :
         #pool = Pool(processes=6)
        #pool.map(foo, range(30))
and 
        #lock = Lock() 
inside the foo () function

and also comment:
        foo(int(argv[2]))
to run this in parallel on the local pc.

'''


def CSS_ch1 (kp, ks, N, x,y,z, SNR, p, e_Seism):

		# e_Seism = list
		# matrices with distances of each seismometer from the others for each coordinate
		
		# I assume every seismometre in the same direction as the others
		e1 = np.array(e_Seism)
		# e2 = np.array([1,0,0])

		mx = np.empty([N,N])
		my = np.empty([N,N])
		mz = np.empty([N,N])
		
		for i in range(0,N):
			for j in range(0,N):
					mx[i,j] = x[i]-x[j]
		
		for i in range(0,N):
			for j in range(0,N):
					my[i,j] = y[i]-y[j]
		
		for i in range(0,N):
			for j in range(0,N):
				mz[i,j] = z[i]-z[j]
		
		dist = np.sqrt(mx**2 + my**2 + mz**2)
		mask = dist == 0
		dist[mask] = 1 #zeros give problems in the division
		
		
		#e12 matrix:
		e12 = np.ones((3,mx.shape[0],mx.shape[1]))
		e12[0,:,:] = mx
		e12[1,:,:] = my 
		e12[2,:,:] = mz
		e12 = e12/dist #normalization
		
		#e1 matrix (equal to e2 since each seismometer is identical to the others)
		
		me1 = np.ones((3,mx.shape[0],mx.shape[1]))
		me1[0,:,:] = me1[0,:,:]*e1[0]
		me1[1,:,:] = me1[1,:,:]*e1[1]
		me1[2,:,:] = me1[2,:,:]*e1[2] 
		
		#me2 = me1
		#matrices for vector's dot operations:
		#(e1.e2) scalar product
		#e1DoTe2 = np.dot(e1,e2)
		e1DoTe2 = 1

		#(e1.e12) & (e2.e12) scalar products
		e1DoTe12 = np.sum(me1*e12,0)
		e2DoTe12 = e1DoTe12; #np.sum(me2*e12,0)
		
		fp = (sp.spherical_jn(0,dist*kp) + sp.spherical_jn(2,dist*kp))*e1DoTe2 - 3.*sp.spherical_jn(2,dist*kp)*e1DoTe12*e2DoTe12	
		fp[mask] = 1 + 1/(SNR)**2 #Diagonal elements are 1 : see notes at p .103
										            
		fs = (sp.spherical_jn(0,dist*ks) - 0.5*sp.spherical_jn(2,dist*ks))*e1DoTe2 + (3./2)*sp.spherical_jn(2,dist*ks)*e1DoTe12*e2DoTe12												          																   
		fs[mask] = 1 + 1/(SNR)**2	#Diagonal elements are 1 : see notes at p .103	
		
		#SS:
		Css = p*fp + (1 - p)*fs
		return Css





def CSN_ch1 (kp, ks, N, x,y,z, SNR, p, e_Seism, e_TestMass):

		dist = np.sqrt(x**2 + y**2 + z**2) #test mass in (0,0,0)
		
		# Seismometers vector
		e1 = np.array(e_Seism)
		# mass test vector always along x
		e2 = np.array(e_TestMass)
		
		#e12 vector 
		e12 = np.ones((3,N))
		e12[0,:] = x
		e12[1,:] = y 
		e12[2,:] = z
		e12 = e12/dist #normalization
		
		#e1 & e2 matrices
		me1 = np.ones((3,N))
		me1[0,:] = me1[0,:]*e1[0]
		me1[1,:] = me1[1,:]*e1[1]
		me1[2,:] = me1[2,:]*e1[2] 

		me2 = np.ones((3,N))
		me2[0,:] = me2[0,:]*e2[0]
		me2[1,:] = me2[1,:]*e2[1]
		me2[2,:] = me2[2,:]*e2[2] 

		
		#(e1.e12) & (e2.e12) scalar products
		e1DoTe12 = np.sum(me1*e12,0)
		e2DoTe12 = np.sum(me2*e12,0)
		
		e1DoTe2 = np.sum(e1*e2)
		
		fp = (sp.spherical_jn(0,dist*kp) + sp.spherical_jn(2,dist*kp))*e1DoTe2 - 3.*sp.spherical_jn(2,dist*kp)*e1DoTe12*e2DoTe12	
		fs = (sp.spherical_jn(0,dist*ks) - 0.5*sp.spherical_jn(2,dist*ks))*e1DoTe2 + (3./2)*sp.spherical_jn(2,dist*ks)*e1DoTe12*e2DoTe12												          																   
		
		#SN:
		Csn = (2/3)*p*fp - (1/3)*(1-p)*fs
		return Csn


def CNN_ch1 (p):
		#NN:
		Cnn = 1/9*(3*p + 1)
		return Cnn






def Residual (state, N, freq, SNR, p):
		        
		kp = 2*np.pi*freq/6000 #velocity for p-waves 6000 m/s
		ks = 2*np.pi*freq/4000 #velocity for s-waves 4000 m/s
		
		# Directions of seismometrs channel of measurment and mass test channel 
		e_Seism = [1,0,0]
		e_TestMass = [1,0,0]
		
		#coordinate of each seismometer and create matrix of distances between each seismometers
		s = state.reshape(N,3)
		
		x = s[:,0]
		y = s[:,1]
		z = s[:,2]
		
		"""************************* correlation between seismometrs calculation: ******************"""	
		Css = CSS_ch1(kp, ks, N, x,y,z, SNR, p, e_Seism)
		"""****************** correlation between seismometrs and test mass calculation: **************"""
		Csn = CSN_ch1(kp, ks, N, x,y,z, SNR, p, e_Seism, e_TestMass)
		"""****************** correlation of the test mass: **************"""
		Cnn = CNN_ch1(p)
		
		
#		resid = 1-np.dot(Csn,np.linalg.inv(Css)).dot(Csn)/Cnn
		X = linalg.solve(Css, Csn, sym_pos=True, overwrite_a=True)
		resid = 1-np.dot(Csn,X)/Cnn		
		if (resid < 0):
			print("NEGATIVE RESIDUAL", resid)
		
		return np.sqrt(resid)
 
 




def foo(N,hh):#(hh)
		#Parameters definitions
		freq = 10
		SNR = 15
		p = 1/3
		
		
		args = (N, freq, SNR, p)
		MA = {'args': args}
		 
		sigma = 200#m #Normal for choosing starting point
		sigma_vector = [sigma]*N*3 
		sigma_matrix = np.diag(sigma_vector)	
		#np.random.seed(100) #fix the initial state and the subsequent positions (for reproducibility)
		initial_state = np.zeros(N*3)
		initial_state = np.random.multivariate_normal(initial_state.flatten() , sigma_matrix,1)
		
		t0 = clock()
		
		StSz = 10 
		Temp = 0.1
		NITER = int(N*200)
		
		result = basinhopping(Residual, initial_state, niter=NITER, T=Temp, stepsize=StSz, 
						     minimizer_kwargs=MA, disp=True, seed=None, niter_success=int(N*40))
		t1 = clock()

		#lock = Lock()
		filename = 'Results'+str(hh)+'.txt'
		f = open(filename,'a+')
		
		                        
		
		f.write('\n \n \n*************BH-bulk-'+ str(hh) + '*************** \n \n \n' )
		        
		
		f.write(str(result.message))
		f.write('\n')

		f.write('Temperature = ')
		json.dump(Temp, f)	
		f.write('\n')

		f.write('N iter = ')
		json.dump(NITER, f)	
		f.write('\n')

		f.write('Step Size = ')
		json.dump(StSz, f)	
		f.write('\n')

		f.write('p = ')
		json.dump(p, f)
		f.write('\n')
		        
		
		f.write('SNR = ')
		json.dump(SNR, f)
		f.write('\n')
		
		
		f.write('N seismometers = ')
		json.dump(N,f)
		f.write('\n')
		        
		f.write('Time elapsed = ')
		json.dump(t1-t0,f)
		f.write('\n')
		
		f.write('Energy = ')
		json.dump(result.fun,f)
		f.write('\n')
		
		aa = 'FinalState'+str(hh)+' = np.array('
		f.write(aa)
		json.dump(result.x.tolist() , f) 
		f.write(')\n')
		aa = 'FinalState'+str(hh) + ' = FinalState'+str(hh)+'.reshape(N,3)\n'
		f.write(aa)
		aa = 'ax.scatter(FinalState'+str(hh)+'[:,0],FinalState'+str(hh)+'[:,1],FinalState'+str(hh)+'[:,2],c=\'g\', marker=\'o\')\n'
		f.write(aa)
		        
		f.close()       
		
		
		
		






"""*********************************************************   MAIN   ******************************************************************"""


if __name__ == '__main__':
        
        
        #pool = Pool(processes=6)
        #pool.map(foo, range(30))
        foo(int(argv[1]), int(argv[2]))










