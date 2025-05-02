from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.single.general_optimizer import GeneralOptimizerPSO
from pyswarms.backend.topology import Pyramid
from pyswarms.backend.topology import Random
from pyswarms.backend.topology import VonNeumann
from pyswarms.backend.topology import Star
from pyswarms.backend.topology import Ring


import numpy as np
from scipy import special as sp
from scipy import linalg

import json
from sys import argv

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


'''
Limit for cavity negligible respect to wavelength of the seismic field (ka->0). 
Seismometers with 3 measuring channels: x,y,z.
Test mass moving along x

To be run like:
	
	python3 nnOpt_DE_bulk N hh workers
	
N = n° of seismometers
hh = Just a number that give the name to the Results<hh>.txt file in output
workers = workers parameter of the Particle Swarm function (n_processes parameter, to parallelize the optimization)

NOTE1: run multiple times this algorithms to find the best minimum (with hh from 1 to 100 or more...)

'''




def CSS_3ch (kp, ks, N, x,y,z, SNR, p):
        # matrices with distances of each seismometer from the others for each coordinate
        if True:
            # optimizing, set zeros, only one loop, do not calculate diagonal elem but leave zero, antisymmetric part of the matrix
            mx = np.zeros([N,N])            
            my = np.zeros([N,N])
            mz = np.zeros([N,N])

            # triangular matrix without diag(zero!)
            for i in range(0,N):
                for j in range(i+1,N):
                    mx[i,j] = x[i]-x[j]
                    my[i,j] = y[i]-y[j]
                    mz[i,j] = z[i]-z[j]
                
            # remaining part
            mx=-mx.T+mx
            my=-my.T+my
            mz=-mz.T+mz
            #tmp = np.concatenate((mx.reshape(N*N,1),my.reshape(N*N,1), mz.reshape(N*N,1)),axis=1)
            #dist = squareform(pdist(tmp, metric='euclidean'))
            dist = np.sqrt(mx**2 + my**2 + mz**2)
        else:
            mx = pdist(x.reshape(N,1), lambda u, v: (u-v)) 
            my = pdist(y.reshape(N,1), lambda u, v: (u-v)) 
            mz = pdist(z.reshape(N,1), lambda u, v: (u-v))
            dist = squareform(np.sqrt(mx**2 + my**2 + mz**2))
            mx=squareform(mx)
            my=squareform(my)
            mz=squareform(mz)
        
        mask = dist == 0
        dist2 = dist.copy()
        dist2[mask] = 1 #zeros give problems in the division
        
        mx = mx/dist2
        my = my/dist2
        mz = mz/dist2
        
        #matrices for vector's dot operations:
        #3N x 3N matrices
        #(e1.e2) scalar product
        zo = np.zeros((N,N))
        o = np.ones((N,N))
        e1DoTe2 = np.concatenate((np.concatenate((o,zo,zo),1),np.concatenate((zo,o,zo),1),np.concatenate((zo,zo,o),1)),0)
        
        #(e1.e12) & (e2.e12) scalar products
        #3N x 3N matrices
        
        e1DoTe12 = np.concatenate((np.concatenate((mx,mx,mx),1),np.concatenate((my,my,my),1),np.concatenate((mz,mz,mz),1)),0)
        e2DoTe12 = np.concatenate((np.concatenate((mx,mx,mx),0),np.concatenate((my,my,my),0),np.concatenate((mz,mz,mz),0)),1)
        
        tmp = np.concatenate((dist,dist,dist),1)
        dist_m = np.concatenate((tmp,tmp,tmp),0)
        
        #tmp = np.concatenate((mask,mask,mask),1)
        #mask = np.concatenate((tmp,tmp,tmp),0)
        #del tmp
        
        fp = (sp.spherical_jn(0,dist_m*kp) + sp.spherical_jn(2,dist_m*kp))*e1DoTe2 - 3.*sp.spherical_jn(2,dist_m*kp)*e1DoTe12*e2DoTe12        
        # fp[mask] = 0        #Diagonal elements are 0 if the channel are perp. : see notes at p .103                                                                                                   
        dd = np.diag(np.ones(fp.shape[1],dtype=bool))
        fp[dd] = 1 + 1/(SNR)**2 #Diagonal elements are 1 : see notes at p .103
        
        
        fs = (sp.spherical_jn(0,dist_m*ks) - 0.5*sp.spherical_jn(2,dist_m*ks))*e1DoTe2 + (3./2)*sp.spherical_jn(2,dist_m*ks)*e1DoTe12*e2DoTe12                                                                                                                                                                                                                                           
        # fs[mask] = 0        #Diagonal elements are 0 if the channel are perp. : see notes at p .103 
        dd = np.diag(np.ones(fs.shape[1],dtype=bool))
        fs[dd] = 1 + 1/(SNR)**2 #Diagonal elements are 1 : see notes at p .103
        #SS:
        Css = p*fp + (1 - p)*fs
        #diagN = np.diag(N*[1])
        #snr = 1/(SNR**2)*np.concatenate((np.concatenate((zo, diagN, diagN), 1),np.concatenate((diagN, zo, diagN), 1), np.concatenate((diagN, diagN, zo), 1)),0 )
        #Css = Css + snr
        return Css

    


def CSN_corr_funct (kp, ks, N, x,y,z, SNR, p, e_TestMass, d_TM1):

        # It is like having 3N sensors, 3 at each point, each one measuring along x,y,z
        xx  = np.concatenate((x,x,x), axis = 0)
        yy  = np.concatenate((y,y,y), axis = 0)
        zz  = np.concatenate((z,z,z), axis = 0)

        #es matrix
        mes = np.zeros((3,3*N))
        mes[0,0:N] = np.ones(N)
        mes[1,N:2*N] = np.ones(N)
        mes[2,2*N:3*N] = np.ones(N)


        e1 = np.array(e_TestMass)
        #es1 vector 

        es1 = np.ones((3,3*N))
        es1[0,:] = xx - d_TM1*e1[0]
        es1[1,:] = yy - d_TM1*e1[1]
        es1[2,:] = zz - d_TM1*e1[2]

        dist1 = np.sqrt(np.sum(es1**2, 0))
        es1 = es1/dist1 #normalization


        #e1 matrix
        me1 = np.ones((3,3*N))
        me1[0,:] = me1[0,:]*e1[0]
        me1[1,:] = me1[1,:]*e1[1]
        me1[2,:] = me1[2,:]*e1[2]

        #(es.es1) & (e1.es1) scalar products
        #e1 := test mass oscillation direction: e_TestMass

        esDoTes1 = np.sum(mes*es1,0)
        e1DoTes1 = np.sum(me1*es1,0)

        esDoTe1 = np.zeros(3)
        esDoTe1[0] = e1[0] # scalar product with x component of the sensor
        esDoTe1[1] = e1[1] # scalar product with y component of the sensor
        esDoTe1[2] = e1[2] # scalar product with z component of the sensor

        fp_s1 = np.zeros(3*N)
        fs_s1 = np.zeros(3*N)

        fp_s1[0:N] = (sp.spherical_jn(0,dist1[0:N]*kp) + sp.spherical_jn(2,dist1[0:N]*kp))*esDoTe1[0] - 3.*sp.spherical_jn(2,dist1[0:N]*kp)*esDoTes1[0:N]*e1DoTes1[0:N]	
        fs_s1[0:N] = (sp.spherical_jn(0,dist1[0:N]*ks) - 0.5*sp.spherical_jn(2,dist1[0:N]*ks))*esDoTe1[0] + (3./2)*sp.spherical_jn(2,dist1[0:N]*ks)*esDoTes1[0:N]*e1DoTes1[0:N]												          																   

        fp_s1[N:2*N] = (sp.spherical_jn(0,dist1[N:2*N]*kp) + sp.spherical_jn(2,dist1[N:2*N]*kp))*esDoTe1[1] - 3.*sp.spherical_jn(2,dist1[N:2*N]*kp)*esDoTes1[N:2*N]*e1DoTes1[N:2*N]	
        fs_s1[N:2*N] = (sp.spherical_jn(0,dist1[N:2*N]*ks) - 0.5*sp.spherical_jn(2,dist1[N:2*N]*ks))*esDoTe1[1] + (3./2)*sp.spherical_jn(2,dist1[N:2*N]*ks)*esDoTes1[N:2*N]*e1DoTes1[N:2*N]												          																   

        fp_s1[2*N:3*N] = (sp.spherical_jn(0,dist1[2*N:3*N]*kp) + sp.spherical_jn(2,dist1[2*N:3*N]*kp))*esDoTe1[2] - 3.*sp.spherical_jn(2,dist1[2*N:3*N]*kp)*esDoTes1[2*N:3*N]*e1DoTes1[2*N:3*N]	
        fs_s1[2*N:3*N] = (sp.spherical_jn(0,dist1[2*N:3*N]*ks) - 0.5*sp.spherical_jn(2,dist1[2*N:3*N]*ks))*esDoTe1[2] + (3./2)*sp.spherical_jn(2,dist1[2*N:3*N]*ks)*esDoTes1[2*N:3*N]*e1DoTes1[2*N:3*N]												          																   

        return fp_s1, fs_s1

def CSN_END_ch3 (kp, ks, N, x,y,z, SNR, p, e_TestMass, d_TM):

        fp_s1, fs_s1 =  CSN_corr_funct (kp, ks, N, x,y,z, SNR, p, e_TestMass, d_TM)

        #SN:
        Csn = 1/3*(2*p*fp_s1 - (1-p)*fs_s1) #multiplied by (4 pi rho_0 G) S_tot which simplifies with Css and Cnn

        return Csn

def CSN_2IN_ch3 (kp, ks, N, x,y,z, SNR, p, e_TestMass1, e_TestMass2, d_TM1, d_TM2):

        fp_s1, fs_s1 =  CSN_corr_funct (kp, ks, N, x,y,z, SNR, p, e_TestMass1, d_TM1)
        fp_s2, fs_s2 =  CSN_corr_funct (kp, ks, N, x,y,z, SNR, p, e_TestMass2, d_TM2)

        #SN:
        Csn = 1/3*(2*p*(fp_s2 - fp_s1) - (1-p)*(fs_s2 - fs_s1)) #multiplied by (4 pi rho_0 G) S_tot which simplifies with Css and Cnn
        return Csn

def CNN_3ch (p):
        #NN:
        Cnn = 1/9*(3*p + 1) #multiplied by (4 pi rho_0 G)^2 S_tot which simplifies with Csn and Css
        return Cnn

def CNN_2IN_3ch (p, kp, ks, e_TestMass1, e_TestMass2, d_TM1, d_TM2):
        #NN:
        e2DoTe1 = np.dot(e_TestMass1,e_TestMass2)
        e21 = d_TM1*e_TestMass1-d_TM2*e_TestMass2
        dist1 = np.linalg.norm((e21))
        e21 = e21/dist1
        e2DoTe21 = np.dot(e_TestMass2,e21)
        e1DoTe21 = np.dot(e_TestMass1,e21)
        
        fp = (sp.spherical_jn(0,dist1*kp) + sp.spherical_jn(2,dist1*kp))*e2DoTe1 - 3.*sp.spherical_jn(2,dist1*kp)*e2DoTe21*e1DoTe21
        fs = (sp.spherical_jn(0,dist1*ks) - 0.5*sp.spherical_jn(2,dist1*ks))*e2DoTe1 + (3./2)*sp.spherical_jn(2,dist1*ks)*e2DoTe21*e1DoTe21
        Cnn = 1/9*(2*(3*p + 1) - 2*(4*p*fp + (1-p)*fs))  #multiplied by (4 pi rho_0 G)^2 S_tot which simplifies with Csn and Css
        return Cnn


def CSS_svd_3ch(Css):

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
''' ********************************************************************************************************************************** '''




def Residual (state, N, freq, SNR, p):
        
        kp = 2*np.pi*freq/6000 #velocity for p-waves 6000 m/s
        ks = 2*np.pi*freq/4000 #velocity for s-waves 4000 m/s
        
        # Directions of seismometrs channel of measurment and mass test channel 
        e_TestMass1 = np.array([1,0,0])
        e_TestMass2 = np.array([0.5,np.sqrt(3)/2,0])
        d_TM_inx = 64.12 # m
        d_TM_iny = 64.12 # m
        d_TM_endx = 536.35 # m
        d_TM_endy = 536.35 # m
        # print(state)
        #coordinate of each seismometer and create matrix of distances between each seismometers
        n_part = state.shape[0]
        Res_Vec = np.zeros(n_part)
        
        for tt in range(0,n_part):
            s = state[tt,:].reshape(N,3)
            x = s[:,0]
            y = s[:,1]
            z = s[:,2]
            # x = np.array(s[:,0],copy=False,dtype=np.dtype('d'))
            # y = np.array(s[:,1],copy=False,dtype=np.dtype('d'))
            # z = np.array(s[:,2],copy=False,dtype=np.dtype('d'))
        
        
            """************************* correlation between seismometrs calculation: ******************""" 
            Css = CSS_3ch(kp, ks, N, x,y,z, SNR, p)
            """****************** correlation between seismometrs and test mass calculation: **************"""
            Csn_in = CSN_2IN_ch3(kp, ks, N, x,y,z, SNR, p, e_TestMass1, e_TestMass2, d_TM_inx, d_TM_iny)
            Csn_end1 = CSN_END_ch3(kp, ks, N, x,y,z, SNR, p, e_TestMass1, d_TM_endx)
            Csn_end2 = CSN_END_ch3(kp, ks, N, x,y,z, SNR, p, e_TestMass2, d_TM_endy)
            """****************** correlation of the test mass: **************"""
            Cnn_END = CNN_3ch(p)
            Cnn_2IN = CNN_2IN_3ch (p, kp, ks, e_TestMass1, e_TestMass2, d_TM_inx, d_TM_iny)
    
    
            """ ************* RESIDUAL CALCULATION ***********************"""
            # Csn = [Csn_in] 
            Csn = [Csn_in, Csn_end1, Csn_end2]
            # Cnn = [Cnn_2IN] 
            Cnn = [Cnn_2IN, Cnn_END, Cnn_END]
            nn = len(Csn)
            Res_v = np.zeros(nn)
    
            for rr in range(0,nn):
                X = linalg.solve(Css, Csn[rr], sym_pos=True, overwrite_a=True)
                resid = 1-np.dot(Csn[rr],X)/Cnn[rr]
                if (resid < 0):
                    print("NEGATIVE RESIDUAL", resid, "-- rr=", rr)
                    Css_svd = CSS_svd_3ch(Css)
                    resid = 1 - np.dot(Csn[rr].conjugate(),np.dot(Css_svd,Csn[rr]))/Cnn[rr]
                    # print('residual ', resid)
                Res_v[rr] = resid
                
            # Res_Vec[tt] = np.sum(Res_v) 
            Res_Vec[tt] = np.max(Res_v)
        return Res_Vec



"""*********************************************************   MAIN   ******************************************************************"""



def foo(N=10, hh=0, worker=1):
                
        #Parameters definitions
        freq = 1

        #Signal to Noise Ratio
        SNR = 15

        #Mixing ratio (p*100% of S and (1-p)*100% of P; p is in the interval [0,1])
        p = 0.2
        
        


        # **************************** DIFFERENTIAL EVOLUTION ALGORITHM *************************************
        print('starting proc ...')
   	# initiate the optimizer
        if N<7:
            x_min = np.tile([-50.,-50.,-150.], N)
            x_max = np.tile([500.,500.,150.], N)
            bounds = (x_min, x_max)
            npart = 40000
            # init = np.tile([100*np.sqrt(3),100.,0.], (npart,N))
            niter = 1000
            options = {'c1': 0.9, 'c2': 1.5, 'w': 0.4, 'k': 20, 'p':2}
            optimizer = GeneralOptimizerPSO(n_particles=npart, dimensions=3*N, options=options, bounds=bounds, ftol = 1e-50,ftol_iter = 20, topology=Ring())
                                            
        else:
            x_min = np.tile([-100.,-300.,-300.], N)
            x_max = np.tile([700.,700.,300.], N)
            bounds = (x_min, x_max)
            npart = 80
            niter = 3000
            options = {'c1': 1.5, 'c2': 1.9, 'w': 0.1, 'k': N*4}#, 'p':1}
            optimizer = GeneralOptimizerPSO(n_particles=npart, dimensions=3*N, options=options, bounds=bounds, ftol = 1e-60,ftol_iter = 30, topology=Random())

        Final_State = optimizer.optimize(Residual, niter, n_processes=worker, N=N, freq=freq, SNR=SNR, p=p )
        
        fun = Final_State[0]
        best_x = Final_State[1]
        
        filename = 'Results'+str(hh)+'.txt'
        f = open(filename,'a+') 

        # e_TestMass1 = np.array([1,0,0])
        # e_TestMass2 = np.array([0.5,np.sqrt(3)/2,0])
        d_TM_inx = 64.12 # m
        d_TM_iny = 64.12 # m
        d_TM_endx = 536.35 # m
        d_TM_endy = 536.35 # m 

        f.write('\n \n \n## *************DE-bulk-'+ str(hh) + '*************** ##\n \n \n' )
        
        f.write('import numpy as np\n')
        f.write('import matplotlib.pyplot as plt\n')
        f.write('fig = plt.figure()\n')
        f.write('ax = fig.add_subplot(111, projection=\'3d\')\n')

        f.write('p = ')
        json.dump(p, f)
        f.write('\n')

        f.write('SNR = ')
        json.dump(SNR, f)
        f.write('\n')

        f.write('N = ')
        json.dump(N,f)
        f.write('\n')

        f.write('f = ')
        json.dump(freq,f)
        f.write('\n')

        f.write('Energy = ')
        # remember to take the root, skipped in the optimizing process
        json.dump(np.sqrt(fun),f)
        f.write('\n')
        
        f.write('e2 = ')
        json.dump(d_TM_iny,f)
        f.write('*np.array([0.5,np.sqrt(3)/2,0])')
        f.write('\ne1 = ')
        json.dump(d_TM_inx, f)
        f.write('*np.array([1,0,0])\n')

        f.write('e3 = ')
        json.dump(d_TM_endy,f)
        f.write('*np.array([0.5,np.sqrt(3)/2,0])')
        f.write('\ne4 = ')
        json.dump(d_TM_endx, f)
        f.write('*np.array([1,0,0])\n')


        aa = 'FinalState'+str(hh)+' = np.array('
        f.write(aa)
        json.dump(best_x.tolist() , f) 
        f.write(')\n')
        aa = 'FinalState'+str(hh) + ' = FinalState'+str(hh)+'.reshape(N,3)\n'
        f.write(aa)
        aa = 'ax.scatter(FinalState'+str(hh)+'[:,0],FinalState'+str(hh)+'[:,1],FinalState'+str(hh)+'[:,2],c=\'g\', marker=\'o\')\n'
        f.write(aa)
        aa = 'ax.scatter(e1[0],e1[1],e1[2],c=\'r\', marker=\'o\')\n'
        f.write(aa)
        aa = 'ax.scatter(e2[0],e2[1],e2[2],c=\'r\', marker=\'o\')\n'
        f.write(aa)
        aa = 'ax.scatter(e3[0],e3[1],e3[2],c=\'r\', marker=\'o\')\n'
        f.write(aa)
        aa = 'ax.scatter(e4[0],e4[1],e4[2],c=\'r\', marker=\'o\')\n'
        f.write(aa)
        
        aa = 'plt.plot([0,e4[0]], [0,e4[1]], \'--\', c=\'k\')\nplt.plot([0, e3[0]], [0, e3[1]], \'--\', c = \'k\')\n'
        f.write(aa)
        f.write('plt.show()\n')
        aa = 'plt.xlabel(\'x\')\n'
        f.write(aa)
        aa = 'plt.ylabel(\'y\')\n'
        f.write(aa)
        
        f.close()       
        

        
#def main(N=15, hh=0, ww=8):

if __name__ == '__main__':
        
        N = int(argv[1]) #number of seismometers
        hh = int(argv[2]) #job identifier 
        ww = int(argv[3]) #workers
        #pool = Pool(processes=6)
        #pool.map(foo, range(30))
    
        #Number of seismometers (At the end it's like having N*3 seismometers since each seismometer is composed by three channels (x,y,z): like having 3 seismometers in N positions)

        foo(N, hh, ww)







