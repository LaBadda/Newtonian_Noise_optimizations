from scipy.optimize import differential_evolution
import numpy as np
from scipy import special as sp
from scipy import linalg

import pickle
import json
import time 
from sys import argv


'''

Hypotesis with only rayleigh field isotropic and homogeneus 

Test mass moving along x

To be run like:
	
	python3 nnOpt_DE_Ryleigh N hh
	
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

def CSS_ch1 (kp, N, x,y, SNR):
	
		# e_Seism = list
		# matrices with distances of each seismometer from the others for each coordinate
		
		mx = np.empty([N,N])
		my = np.empty([N,N])

		for i in range(0,N):
			for j in range(0,N):
					mx[i,j] = x[i]-x[j]
		
		for i in range(0,N):
			for j in range(0,N):
					my[i,j] = y[i]-y[j]

		dist = np.sqrt(mx**2 + my**2)

		Css = sp.j0(kp*dist)
		dd = np.diag(np.ones(Css.shape[1],dtype=bool))
		Css[dd] = Css[dd] + 1/((SNR)**2)
		return [Css,dist]





def CSN_ch1 (kp, N, x,y, SNR):

		dist = np.sqrt(x**2 + y**2) #test mass in (0,0,0)
		
		
#		es = np.array([x,y])/dist;
#		z = np.zeros((1,N))
#		o = np.ones((1,N))
#		et = np.concatenate((o,z),0)
#
#		phi = np.sum(es*et,0);

		Cos_phi = x/dist
		Csn = Cos_phi*sp.j1(kp*dist)
		return [Csn,dist]


#def CNN_ch1 (p):
#		#NN:
#		Cnn = 0.5
#		return Cnn






def Residual (state, N, freq, SNR):
		        
		kp = 2*np.pi*freq/100 #velocity for p-waves 6000 m/s

		#coordinate of each seismometer and create matrix of distances between each seismometers
		s = state.reshape(N,2)
		
		x = s[:,0]
		y = s[:,1]
		
		"""************************* correlation between seismometrs calculation: ******************"""	
		Css = CSS_ch1(kp, N, x,y, SNR)
		dss = Css[1]
		Css = Css[0]
		"""****************** correlation between seismometrs and test mass calculation: **************"""
		Csn = CSN_ch1(kp, N, x,y, SNR)
		dsn = Csn[1]
		Csn = Csn[0]
		"""****************** correlation of the test mass: **************"""
		Cnn = 0.5
		
		

#		resid = 1-np.dot(Csn,np.linalg.inv(Css)).dot(Csn)/Cnn
		X = linalg.solve(Css, Csn, sym_pos=True, overwrite_a=True)
		resid = 1-np.dot(Csn,X)/Cnn		

		if (resid < 0):
			print("NEGATIVE RESIDUAL", resid, " --- Css = ", np.linalg.inv(Css), " --- dss = ",dss, " --- Csn = ", Csn, " --- dsn = ", dsn)
		
		return np.sqrt(resid)
 


"""*********************************************************   MAIN   ******************************************************************"""



def foo(N, hh):
         
        
        #Parameters definitions
        freq = 10
        SNR = 10
        #Dimension (2D)
        D = 2
        
        bound = [(-5., 5.)]
        x_bound = bound*N*D
        
        
        
        Res_Par = (N, freq, SNR)

#        t0 = time.time()
        Final_State = differential_evolution(Residual, x_bound, Res_Par, disp=True, maxiter=4500, popsize=35, tol = 1e-7)#,  recombination=0.5, mutation=(0.5, 1), strategy='best2bin')
#        t1 = time.time()
#        filehandler = open('DE_FS_N' + str(N) + '_' + str(hh) + '.obj', 'wb') 
#        pickle.dump(Final_State, filehandler)
#        filehandler.close()	

        #lock = Lock()  

        filename = 'Results'+str(hh)+'.txt'
        f = open(filename,'a+')

        filename = 'En_'+str(hh)+'.txt'
        g = open(filename,'a+')


#        f.write('\n \n \n*************DE-Rayleigh-'+ str(hh) + '*************** \n \n \n' )



        json.dump(Final_State.fun,g)
        g.write('\n')

        f.write('Energy = ')
        json.dump(Final_State.fun,f)
        f.write('\n')

        aa = 'FinalState'+str(hh)+' = np.array('
        f.write(aa)
        json.dump(Final_State.x.tolist() , f) 
        f.write(')\n')
        aa = 'FinalState'+str(hh) + ' = FinalState'+str(hh)+'.reshape(N,2)\n'
        f.write(aa)
        aa = 'ax.scatter(FinalState'+str(hh)+'[:,0],FinalState'+str(hh)+'[:,1],c=np.ones((N))*Energy,cmap=\'RdBu\',vmin=Emin,vmax=Emax, marker=\'o\')\n'
        f.write(aa)
        
        f.close()
        g.close()



if __name__ == '__main__':
        
#        t0 = time.time()
#        pool = Pool(processes=6)
#        pool.map(foo, range(3))
#        argv[1] = 4#Number of seismometers

        foo(int(argv[1]), int(argv[2]))
#        t1 = time.time()
#        print(t1-t0)