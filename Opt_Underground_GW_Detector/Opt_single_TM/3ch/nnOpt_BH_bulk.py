from scipy import special as sp
from scipy.optimize import basinhopping
import numpy as np
from scipy import linalg

import json
from time import clock
from sys import argv

'''

Limit for cavity negligible respect to wavelength of the seismic field (ka->0). 
Seismometers with 3 measuring channels: x,y,z.
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
        foo(int(argv[1]), int(argv[2]))
to run this in parallel on the local pc.

'''




def CSS_3ch (kp, ks, N, x,y,z, SNR, p):
        # matrices with distances of each seismometer from the others for each coordinate
        
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
        
        mx = mx/dist
        my = my/dist
        mz = mz/dist
        
        #matrices for vector's dot operations:
        #(e1.e2) scalar product
        zo = np.zeros((N,N))
        o = np.ones((N,N))
        e1DoTe2 = np.concatenate((np.concatenate((o,zo,zo),1),np.concatenate((zo,o,zo),1),np.concatenate((zo,zo,o),1)),0)
        
        #(e1.e12) & (e2.e12) scalar products
        e1DoTe12 = np.zeros((3*N,3*N))
        
        e1DoTe12 = np.concatenate((np.concatenate((mx,mx,mx),1),np.concatenate((my,my,my),1),np.concatenate((mz,mz,mz),1)),0)
        e2DoTe12 = np.concatenate((np.concatenate((mx,mx,mx),0),np.concatenate((my,my,my),0),np.concatenate((mz,mz,mz),0)),1)
        
        tmp = np.concatenate((dist,dist,dist),1)
        dist = np.concatenate((tmp,tmp,tmp),0)
        
        tmp = np.concatenate((mask,mask,mask),1)
        mask = np.concatenate((tmp,tmp,tmp),0)
        del tmp
        
        fp = (sp.spherical_jn(0,dist*kp) + sp.spherical_jn(2,dist*kp))*e1DoTe2 - 3.*sp.spherical_jn(2,dist*kp)*e1DoTe12*e2DoTe12
        fp[mask] = 0      # Diagonal elements are 0 if the channel are perp. : see notes at p .103

        dd = np.diag(np.ones(fp.shape[1],dtype=bool))
        fp[dd] = 1 + 1/(SNR)**2 #Diagonal elements are 1 : see notes at p .103
        
        
        fs = (sp.spherical_jn(0,dist*ks) - 0.5*sp.spherical_jn(2,dist*ks))*e1DoTe2 + (3./2)*sp.spherical_jn(2,dist*ks)*e1DoTe12*e2DoTe12
        fs[mask] = 0      # Diagonal elements are 0 if the channel are perp. : see notes at p .103 

        dd = np.diag(np.ones(fs.shape[1],dtype=bool))
        fs[dd] = 1 + 1/(SNR)**2 #Diagonal elements are 1 : see notes at p .103

        #SS:
        Css = p*fp + (1 - p)*fs
        return Css

    
def CSN_3ch (kp, ks, N, x,y,z, SNR, p):
        dist = np.sqrt(x**2 + y**2 + z**2) #test mass in (0,0,0)
        dist = np.concatenate((dist,dist,dist))
        #e12 vector 
        
        #(e1.e12) & (e2.e12) scalar products
        #e1 := test mass oscillation direction: (1,0,0)
        
        e1DoTe12 = np.concatenate((x,x,x))/dist
        e2DoTe12 = np.concatenate((x,y,z))/dist
        e1DoTe2 = np.concatenate((np.ones(N),np.zeros(2*N)))
        
        fp = (sp.spherical_jn(0,dist*kp) + sp.spherical_jn(2,dist*kp))*e1DoTe2 - 3.*sp.spherical_jn(2,dist*kp)*e1DoTe12*e2DoTe12        
        fs = (sp.spherical_jn(0,dist*ks) - 0.5*sp.spherical_jn(2,dist*ks))*e1DoTe2 + (3./2)*sp.spherical_jn(2,dist*ks)*e1DoTe12*e2DoTe12                                                                                                                                                                                                                                           
        
        #SN:
        Csn = (2/3)*p*fp - (1/3)*(1-p)*fs
        return Csn


def CNN_3ch (p):
        #NN:
        Cnn = 1/9*(3*p + 1)
        return Cnn




''' ********************************************************************************************************************************** '''




def Residual (state, N, freq, SNR, p):

        fstart = 3
        F = np.linspace(fstart, freq,3)#(fstart, freq, freq-fstart+1) #freq : fstart, fstart +1, fstart +2 ...
        resid = np.zeros(len(F))

        for i in range(0, len(F)):

           kp = 2*np.pi*F[i]/6000 #velocity for p-waves 6000 m/s
           ks = 2*np.pi*F[i]/4000 #velocity for s-waves 4000 m/s
        
           #coordinate of each seismometer and create matrix of distances between each seismometers
           s = state.reshape(N,3)

           x = s[:,0]
           y = s[:,1]
           z = s[:,2]
        
           """************************* correlation between seismometrs calculation: ******************""" 
           Css = CSS_3ch(kp, ks, N, x,y,z, SNR, p)
           """****************** correlation between seismometrs and test mass calculation: **************"""
           Csn = CSN_3ch(kp, ks, N, x,y,z, SNR, p)
           """****************** correlation of the test mass: **************"""
           Cnn = CNN_3ch(p)


           """ ************* RESIDUAL CALCULATION ***********************"""
#           resid[i] = 1-np.dot(Csn,np.linalg.inv(Css)).dot(Csn)/Cnn
           X = linalg.solve(Css, Csn, sym_pos=True, overwrite_a=True)
           resid[i] = 1-np.dot(Csn,X)/Cnn
#           if (resid < 0):
#               print("NEGATIVE RESIDUAL", resid)

#           r = np.sqrt(resid)
        return np.max(resid)







def foo(N,hh):


                #Parameters definitions
                freq = 10

                #Signal to Noise Ratio
                SNR = 15

                #Mixing ratio (p*100% of S and (1-p)*100% of P; p is in the interval [0,1])
                p = 1/3

                args = (N, freq, SNR, p)
                MA = {'args': args}
                 
                sigma = 200  #m #Normal for choosing starting point
                sigma_vector = [sigma]*3*N 
                sigma_matrix = np.diag(sigma_vector)    
                #np.random.seed(100) #fix the initial state and the subsequent positions (for reproducibility)
                initial_state = np.zeros(3*N)
                initial_state = np.random.multivariate_normal(initial_state.flatten() , sigma_matrix,1)

                t0 = clock()
                
                StSz = 10 
                Temp = 0.01
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










