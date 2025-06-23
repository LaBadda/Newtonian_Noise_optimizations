import sys
import numpy as np

from scipy import linalg
from scipy import special as sp
from matplotlib import pyplot as plt
from pyswarms.single.global_best import GlobalBestPSO


class my_PSO():
  def __init__(self, xm, xM, ym, yM, x_TM1, y_TM1, x_TM2, y_TM2):
    
    self.xm = xm
    self.xM = xM
    self.ym = ym
    self.yM = yM
    self.x_TM1 = x_TM1
    self.y_TM1 = y_TM1
    self.x_TM2 = x_TM2
    self.y_TM2 = y_TM2

  def check_rect(self,x,y,x_left, x_right, y_low, y_up):
    if ((x > x_left) & (x < x_right) & (y > y_low) & (y < y_up)):
      return True
    else:
      return False

  def CSS (self, kp, N, x,y, SNR):
    
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


  def CSN (self, kp, N, x,y, x_TM1, y_TM1, x_TM2, y_TM2, TM = 'x'):
    h = np.zeros(N)
    for uu in range(0,N):
        if (self.check_rect(x[uu],y[uu], -1.75, 1.75, 1.75, 8.95) |
            self.check_rect(x[uu],y[uu], -8.15, 12.90, -1.75, 1.75) | 
            self.check_rect(x[uu],y[uu], -1.75, 1.75, -12.75, -1.75) | 
            self.check_rect(x[uu],y[uu], 1.75, 4.6, -8.5, -1.75) |  #WROOONGG!!!!!!!!!!
            self.check_rect(x[uu],y[uu], 4.6, 7.75, -13.35, -1.75)):
          h[uu] = 4.95
        else:
          h[uu] = 1.1

            
    if (TM == 'x'):
      dist = np.sqrt((x-x_TM1)**2 + (y-y_TM1)**2 + h**2) #test mass in (0,0,0)
      Trigonometric_phi = -(x-x_TM1)/dist #cos in the Virgo direction
    elif (TM == 'y'):
      dist = np.sqrt((x-x_TM2)**2 + (y-y_TM2)**2 + h**2) #test mass in (0,0,0)
      Trigonometric_phi = (y-y_TM2)/dist #sin
    else:
      sys.exit("some error message") 
        
    Csn = Trigonometric_phi*sp.j1(kp*dist)
    return [Csn,dist]




  # def Residual (self, state, N, freq, SNR, x_TM1, y_TM1, x_TM2, y_TM2):
              
  #     kp = 2*np.pi*freq/100 

  #     if len(state.shape) == 1:
  #        state = np.expand_dims(state, 0)

  #     #coordinate of each seismometer and create matrix of distances between each seismometers
  #     s = state.reshape(N,2)
      
  #     x = s[:,0]
  #     y = s[:,1]
      
  #     """************************* correlation between seismometrs calculation: ******************"""	
  #     Css = self.CSS(kp, N, x,y, SNR)
  #     dss = Css[1]
  #     Css = Css[0]
  #     """****************** correlation between seismometrs and test mass calculation: **************"""
  #     Csnx = self.CSN(kp, N, x,y, SNR, x_TM1, y_TM1, x_TM2, y_TM2, TM = 'x') 
  #     Csny = self.CSN(kp, N, x,y, SNR, x_TM1, y_TM1, x_TM2, y_TM2, TM = 'y')
  #     Csn = Csnx[0] - Csny[0]

  #     """****************** correlation of the test mass: **************"""
  #     Cnn = 2*0.5
      
  #     X = linalg.solve(Css, Csn, sym_pos=True, overwrite_a=True)
  #     resid = 1-np.dot(Csn,X)/Cnn		
  #     # resid = 1 -np.dot(Csn.conjugate(),np.dot(Css,Csn))/Cnn
  #     if (resid < 0):
  #       print("NEGATIVE RESIDUAL", resid, " --- Css = ", np.linalg.inv(Css), " --- Csn = ", Csn, " --- dss = ", dss)      
  #       # print("NEGATIVE RESIDUAL", resid, " --- dx = ",Csnx[1], " --- dy = ", Csny[1])      
  #     return np.sqrt(resid)
  #     # return Csn



  def Residual_pso (self, state, N, freq, SNR, x_TM1, y_TM1, x_TM2, y_TM2):
              
      kp = 2*np.pi*freq/100 

      if len(state.shape) == 1:
         state = np.expand_dims(state, 0)

      n_part = state.shape[0]
      Res_Vec = np.zeros(n_part)

      for tt in range(0,n_part):  
        #coordinate of each seismometer and create matrix of distances between each seismometers
        s = state[tt,:].reshape(N,2)
        
        x = s[:,0]
        y = s[:,1]
        
        """************************* correlation between seismometrs calculation: ******************"""	
        Css = self.CSS(kp, N, x,y, SNR)
        dss = Css[1]
        Css = Css[0]
        """****************** correlation between seismometrs and test mass calculation: **************"""
        Csnx = self.CSN(kp, N, x,y, x_TM1, y_TM1, x_TM2, y_TM2, TM = 'x') 
        Csny = self.CSN(kp, N, x,y, x_TM1, y_TM1, x_TM2, y_TM2, TM = 'y')
        Csn = Csnx[0] - Csny[0]
  
        """****************** correlation of the test mass: **************"""
        Cnn = 2*(0.5)
        
        X = linalg.solve(Css, Csn, sym_pos=True, overwrite_a=True)
        resid = -np.dot(Csn,X)
        # resid = 1-np.dot(Csn,X)/Cnn		
        # resid = 1 -np.dot(Csn.conjugate(),np.dot(Css,Csn))/Cnn
        # if (resid < 0):
        #   print("NEGATIVE RESIDUAL", resid, )#" --- Css = ", np.linalg.inv(Css), " --- Csn = ", Csn, " --- dss = ", dss)      
          # print("NEGATIVE RESIDUAL", resid, " --- dx = ",Csnx[1], " --- dy = ", Csny[1])      
        Res_Vec[tt] = resid
      return Res_Vec
      # return Csn


if (__name__ == '__main__'):
  xm = -19.0
  xM = 14.8
  ym = -10.1
  yM = 17.9
  x_TM1 = -5.6
  y_TM1 = 0
  x_TM2 = 0
  y_TM2 = 5.8
  PSO = my_PSO(xm, xM, ym, yM, x_TM1, y_TM1, x_TM2, y_TM2)

  freq = 15
  SNR = 10
  N = 1

  x0 = np.linspace(xm, xM, 80)
  y0 = np.linspace(ym, yM, 80)
  X, Y = np.meshgrid(x0, y0)
  x = X.flatten()
  y = Y.flatten()
  state = np.stack((x,y), axis = 1)

  Res = np.zeros(len(x))

  c = 0
  for state_i in state:
    Res[c] = PSO.Residual_pso(state_i, N, freq, SNR, x_TM1, y_TM1, x_TM2, y_TM2)
    c+=1



  plt.close()
  fig = plt.figure(8, figsize=(16, 9))
  #ax = fig.get_axes() 

  cax = plt.contourf(X,Y, Res.reshape(X.shape), 200, cmap = 'hot_r')#, vmin = vmin, vmax = vmax)	
  R15min = np.min(Res[:])	
  R15max = np.max(Res[:])	
  cbar = fig.colorbar(cax, ticks=[R15min, (R15max+R15min)/2, R15max], orientation='vertical')
  
  plt.scatter(0,0, marker='*',c='white', edgecolor='k', s=500)
  plt.scatter(x_TM1, y_TM1, marker='*',c='white', edgecolor='k', s=500)
  plt.scatter(x_TM2, y_TM2, marker='*',c='white', edgecolor='k', s=500)



  col = 'k'
  plt.plot((-1.75, 1.75), (8.95, 8.95), c=col)
  plt.plot((-1.75, -1.75), (8.95, 1.75), c=col)
  plt.plot((1.75, 1.75), (8.95, 1.75), c=col)
  plt.plot((-8.15, -1.75), (1.75, 1.75), c=col)
  plt.plot((-8.15, -1.75), (-1.75, -1.75), c=col)
  plt.plot((-8.15, -8.15), (1.75, -1.75), c=col)
  plt.plot((-1.75, -1.75), (-1.75, -12.75), c=col)#@@@@@@@@@@@@@@@@@@@@@@@@@@@
  plt.plot((-1.75, 1.75), (-12.75, -12.75), c=col)#@@@@@@@@@@@@@@@@@@@@@@@@@@@
  plt.plot((1.75, 1.75), (-8.5, -12.75), c=col)
  plt.plot((1.75, 4.6), (-8.5, -8.5), c=col)
  plt.plot((4.6, 4.6), (-8.5, -13.35), c=col)
  plt.plot((4.6, 4.6), (-8.5, -13.35), c=col)
  plt.plot((4.6, 7.75), (-11.6, -13.35), c=col)
  plt.plot((7.75, 7.75), (-11.6, -1.75), c=col)
  plt.plot((7.75, 12.9), (-1.75, -1.75), c=col)
  plt.plot((12.9, 12.9), (1.75, -1.75), c=col)
  plt.plot((12.9, 1.75), (1.75, 1.75), c=col)
  plt.plot((1.75, 1.75), (1.75, 8.95), c=col)
  
  # plt.scatter((xf), (yf), marker = '^',c='y')
  plt.ylim((ym,yM))
  plt.xlim((xm,xM))
  
  ns = 2
  n = 1
  c = 0
  E = [None]*1
  best_xy = [None]*1
  for i in [ns]:#[1,2,3,4,5,6,7,8,9]:
    npart = 60
    niter = i*3500
    
    N = i#2
    x_max = np.tile([xM - 0.01, yM - 0.01], N)
    x_min = np.tile([xm + 0.01, ym + 0.01], N)
    bounds = (x_min, x_max)
    options = {'c1': 0.5, 'c2': 0.9, 'w': 0.9}
    optimizer = GlobalBestPSO(n_particles=npart, dimensions=2*N, options=options, bounds=bounds)
    
    Final_State = optimizer.optimize(PSO.Residual_pso, niter, n_processes=4, N = N, freq = freq, SNR = SNR, x_TM1 =x_TM1, y_TM1 = y_TM1, x_TM2 = x_TM2, y_TM2 =y_TM2)
    cost_history = optimizer.cost_history
    # Final_State = Final_State + (options, {'npart': npart, 'niter':niter}, cost_history)
    best_xy[c] = Final_State[1]
    E[c] = Final_State[0]
    c +=1 
  
    # if (best_xy.shape) == 1:
    #   plt.scatter(best_xy[0],best_xy[1], s = 100, c='tab:pink', marker = 'D', edgecolor='k')
    # else:
    #   best_xy = best_xy.reshape(N,2)
    #   plt.scatter(best_xy[:,0],best_xy[:,1], s = 100, c='tab:pink', marker = 'D', edgecolor='k')
  # print('Residual:     ', E)


  # fig = plt.figure(10, figsize=(16, 9))
  # plt.plot([1,2,3,4,5,6,7,8,9], np.array(E)/min(E), linewidth = 3, label = 'iso+homo')

#%%
  fig = plt.figure(8, figsize=(16, 9))
  for i in [ns-1]:#range(0,8):
    N = i+1
    ii = np.array(best_xy).reshape(N,2)
    # ii = best_xy[i].reshape(N,2)
    plt.scatter(ii[:,0],ii[:,1], s = 100, marker = 'D', edgecolor='k', label = str(N))
    
  plt.scatter(0,0, marker='*',c='white', edgecolor='k', s=500)
  plt.scatter(x_TM1, y_TM1, marker='*',c='white', edgecolor='k', s=500)
  plt.scatter(x_TM2, y_TM2, marker='*',c='white', edgecolor='k', s=500)



  col = 'k'
  plt.plot((-1.75, 1.75), (8.95, 8.95), c=col)
  plt.plot((-1.75, -1.75), (8.95, 1.75), c=col)
  plt.plot((1.75, 1.75), (8.95, 1.75), c=col)
  plt.plot((-8.15, -1.75), (1.75, 1.75), c=col)
  plt.plot((-8.15, -1.75), (-1.75, -1.75), c=col)
  plt.plot((-8.15, -8.15), (1.75, -1.75), c=col)
  plt.plot((-1.75, -1.75), (-1.75, -12.75), c=col)#@@@@@@@@@@@@@@@@@@@@@@@@@@@
  plt.plot((-1.75, 1.75), (-12.75, -12.75), c=col)#@@@@@@@@@@@@@@@@@@@@@@@@@@@
  plt.plot((1.75, 1.75), (-8.5, -12.75), c=col)
  plt.plot((1.75, 4.6), (-8.5, -8.5), c=col)
  plt.plot((4.6, 4.6), (-8.5, -13.35), c=col)
  plt.plot((4.6, 4.6), (-8.5, -13.35), c=col)
  plt.plot((4.6, 7.75), (-11.6, -13.35), c=col)
  plt.plot((7.75, 7.75), (-11.6, -1.75), c=col)
  plt.plot((7.75, 12.9), (-1.75, -1.75), c=col)
  plt.plot((12.9, 12.9), (1.75, -1.75), c=col)
  plt.plot((12.9, 1.75), (1.75, 1.75), c=col)
  plt.plot((1.75, 1.75), (1.75, 8.95), c=col)
  
  # plt.scatter((xf), (yf), marker = '^',c='y')
  plt.ylim((ym,yM))
  plt.xlim((xm,xM))
  plt.legend()
