# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:07:22 2019

@author: Francesca
"""
import pickle 
import numpy as np
from scipy import interpolate as I
from matplotlib import pyplot as plt
import sys
from matplotlib import rcParams
rcParams.update({'font.size': 25})	

N = 30**4
CSS = np.zeros(N,dtype = complex)
x1 = np.zeros(N)
y1 = np.zeros(N)
x2 = np.zeros(N)
y2 = np.zeros(N)

P = 'C:/Users/Francesca/Desktop/Results_and_kriging/Reg_out10/'
dic = {'indx':'_indx.obj','CSS':'_CSS.obj','x1':'_x1.obj', 'y1':'_y1.obj', 'x2':'_x2.obj', 'y2':'_y2.obj'}

for i in range(0,10):
	for j in range(0,10):
		
		pickle_in = open(P+'Regular'+str(i)+str(j)+dic['indx'],"rb")
		indx = pickle.load(pickle_in)
		pickle_in.close()
		indx = indx.astype(int) 
#		print(i,j)
		
		pickle_in = open(P+'Regular'+str(i)+str(j)+dic['CSS'],"rb")
		tmp = pickle.load(pickle_in)
		CSS[indx] = tmp
		pickle_in.close()

		pickle_in = open(P+'Regular'+str(i)+str(j)+dic['x1'],"rb")
		tmp = pickle.load(pickle_in)
		x1[indx] = tmp 
		pickle_in.close()

		pickle_in = open(P+'Regular'+str(i)+str(j)+dic['y1'],"rb")
		tmp = pickle.load(pickle_in)
		y1[indx] = tmp 
		pickle_in.close()

		pickle_in = open(P+'Regular'+str(i)+str(j)+dic['x2'],"rb")
		tmp = pickle.load(pickle_in)
		x2[indx] = tmp 
		pickle_in.close()

		pickle_in = open(P+'Regular'+str(i)+str(j)+dic['y2'],"rb")
		tmp = pickle.load(pickle_in)
		y2[indx] = tmp 
		pickle_in.close()

X = np.unique(x1)
Y = np.unique(y1)

##########check
Xm1, Ym1, Xm2, Ym2 = np.meshgrid(X,Y,X,Y, indexing='ij')

a = x1.reshape(Xm1.shape)
print(np.array_equal(a,Xm1))

a = y1.reshape(Xm1.shape)
print(np.array_equal(a,Ym1))

a = x2.reshape(Xm1.shape)
print(np.array_equal(a,Xm2))

a = y2.reshape(Xm1.shape)
print(np.array_equal(a,Ym2))
###############
Xmf1 = Xm1.flatten()
Ymf1 = Ym1.flatten()
Xmf2 = Xm2.flatten()
Ymf2 = Ym2.flatten()

CSSre = np.real(CSS)
CSSregrid = CSSre.reshape(Xm1.shape)
#from scipy import linalg
#invCss = np.real(linalg.inv((CSS.reshape(900,900))))
#CSSregrid = invCss.flatten().reshape(Xm1.shape) 


Ipmio = np.stack((x1,y1,x2,y2), axis = 1)
InterpRe = I.RegularGridInterpolator((X,Y,X,Y), CSSregrid, method='linear')#, bounds_error=False, fill_value= 1)
#InterpReN = I.NearestNDInterpolator(Ipmio, CSSre)
#%%

filehandler = open(P+'CSS_30x.obj', 'wb') 
pickle.dump(CSS, filehandler)
filehandler.close()

filehandler = open(P+'x1_30x.obj', 'wb') 
pickle.dump(x1, filehandler)
filehandler.close()

filehandler = open(P+'y1_30x.obj', 'wb') 
pickle.dump(y1, filehandler)
filehandler.close()

filehandler = open(P+'x2_30x.obj', 'wb') 
pickle.dump(x2, filehandler)
filehandler.close()

filehandler = open(P+'y2_30x.obj', 'wb') 
pickle.dump(y2, filehandler)
filehandler.close()



#%% PLOT 

coords = np.array([[-2996.1855, 3.3109, -3.4011],[-2998.6876, 7.1706, -3.4071],[-2999.3596, 3.2421, -3.4008],[-3003.9149, 7.8757, -3.4227],[-3003.0811, 3.2749, -3.4150],[-3005.9728, 7.7190, -3.4288],[-3006.2434, 3.3279, -3.4237],[-3010.3192, 7.9888, -3.4437],[-3012.1375, 3.3488, -3.4299],[-3014.7164, 7.5541, -3.4490],[-3017.8701, 4.2944, -3.4613],[-3014.9770, -0.0874, -3.4426],[-3017.9104, -4.0994, -3.4595],[-3015.1017, -7.3702, -3.4573],[-3013.7171, -3.1819, -3.4287],[-3011.1778, -6.0311, -3.4400],[-3009.9611, -3.1498, -3.4302],[-3005.5196, -6.1313, -3.4285],[-3005.9568, -3.1540, -3.4234],[-3003.6825, -6.6868, -3.4195],[-3000.1694, -3.4201, -3.4149],[-2999.4935, -5.4172, -3.4146],[-2996.6250, -3.3377, -3.4085],[-2999.3105, 2.7366, -3.3156],[-3002.9757, 2.6436, -3.3247],[-3006.8761, 2.7944, -3.3304],[-3010.9353, 2.8204, -3.3416],[-3013.8210, 2.3477, -3.3491],[-3013.6882, -2.6371, -3.3435],[-3011.2803, -2.5983, -3.3446],[-3006.4946, -2.6779, -3.3386],[-3002.6733, -2.5689, -3.3296],[-2999.6753, -2.5736, -3.3223],[-2999.8896, 0.0321, -3.3182],[-3009.4183, 0.0789, -3.3497],[-3013.7110, -0.0613, -3.3462],[-3012.2423, -1.1159, -6.8445],[-3008.4083, 1.4434, -6.8571]])	
Xcoord = coords[:,0:2]	
Xcoord = np.delete(Xcoord,  [34], axis=0)
x0 = -3005.5847
y0 = 0.0312
h = np.average(coords[0:21,2]) - (-2.2233)
#centering around test mass
Xcoord[:,0] = Xcoord[:,0] - x0
Xcoord[:,1] = Xcoord[:,1] - y0



Npoints = 100 
de = 0
xi = np.linspace(X.min()+de,X.max()-de,Npoints)
yi = np.linspace(Y.min()+de,Y.max()-de,Npoints)

Xim, Yim = np.meshgrid(xi, yi, indexing='xy')

Xi = Xim.flatten()
Yi = Yim.flatten()

#np.random.seed(666)
#xr = np.random.uniform(Xcoord[:,0].min(), Xcoord[:,0].max())
#yr = np.random.uniform(Xcoord[:,1].min(), Xcoord[:,1].max())
#i=1
#xr= xy[i][0]
#yr= xy[i][1]

#id_seism = 27
#Xref = Xcoord[id_seism,0]
#Yref = Xcoord[id_seism,1]
#Xs = np.ones(Yi.shape)*Xref
#Ys = np.ones(Yi.shape)*Yref
XY = np.array([[-8.1, 4]])
#XY = np.array([[-8, 2.6],[-12, -7], [9, -7], [-12,7], [9,7], [-10, 0], [8, 0], [-8,5], [6,5], [-8,-5], [6, -5], [0,7], [0,-7], [-8,0], [6,0], [5, 2.6], [5,-2.6], [0,-1], [0,1], [-7, 2], [5, -2]])
#XY = np.array([[-8, 2.6], [-8,2.5], [-7.8, 2.6], [-7.8, 2.5], [-8, 2.3], [-7.8, 2.5]])			   
#			   [-10, 6.5], [8.5, 0], [-3, 6], [-3,-6],  [-11, 1], [9, 3.6], [8, -6], [5, 6], [5, -6], [4, 2], [2, 2.6], [-1,0], [-8, 1], [6, -1], [-5, -1], [4, 2.6], [-5, -2.6], [-8,2.6], [6, -2.6]])

Zi0 = np.zeros((XY.shape[0], len(Xi)))
c = 0
for i in XY:
	
	Xref = i[0]
	Yref = i[1]
	Xs = np.ones(Yi.shape)*Xref
	Ys = np.ones(Yi.shape)*Yref
	
	Ip0 = np.stack((Xi, Yi, Xs, Ys), axis = 1)
#	Ip1 = np.stack((Xs, Ys, Xi, Yi), axis = 1)
	Zi0[c, :] = InterpRe.__call__(Ip0)
#	Zi0N = InterpReN.__call__(Ip0)
	c += 1
	
PP = 'C:/Users/Francesca/OneDrive/PhD/Conferenze-Divulgazione-Scuole/Pisa/VirgoWeek-27-29Jan2020/slidesNN/'	
for i in range(0, XY.shape[0]):
	
	plt.figure(i)
	
	plt.pcolor(xi,yi, np.reshape(Zi0[i,:], Xim.shape))#, vmin=np.min(Zi0[:]) - np.min(Zi0[:])*0.5, vmax=np.max(Zi0[:]) - np.max(Zi0[:])*0.5)
#	plt.contourf(Xim,Yim, np.reshape(Zi0, Xim.shape), 20)
	plt.colorbar()
	
	plt.scatter(Xcoord[:,0],Xcoord[:,1], marker='o', c='w', s=70)
	#plt.plot(xr, yr, 'or', linewidth = 10)
	plt.scatter(XY[i,0],XY[i,1], marker='o',c='r', s=70)
	plt.scatter(0,0, marker='*',c='tab:orange', edgecolor='k', s=500)
	#plt.plot(xy[i][0],xy[i][1], 'or', linewidth = 10)
	#plt.title('Regular grid')
	figManager = plt.get_current_fig_manager()
	figManager.window.showMaximized()
	plt.tight_layout()
#	plt.savefig(PP+'Fig_'+str(i)+'.png', bbox_inches='tight')#, transparent=True)

#%%
#
#plt.figure(2)
#
#
#def Kernel(h, x1, y1):
#	return x1/((h**2 + x1**2 + y1**2)**(3/2))
#
#
#h = np.ones(Yim.shape)*1.5
#msk = ((Xim>-8)&(Xim<6) & (Yim>-2.6)&(Yim<2.6))
#h[msk] = 5
#K2 =  Kernel(h,Xim,Yim)
#
#plt.pcolor(Xim,Yim, K2*np.reshape(Zi0, Xim.shape))#, 80)
#plt.colorbar()
#
#plt.plot(Xcoord[:,0],Xcoord[:,1], 'wo')
##plt.plot(xr, yr, 'or', linewidth = 10)
#plt.plot(Xref,Yref, 'or', linewidth = 10)
##plt.plot(xy[i][0],xy[i][1], 'or', linewidth = 10)
#plt.title('Regular grid')
#
#plt.figure(89)
#plt.pcolor(Xim,Yim,K2)
#
#

#%%
#plt.figure()
#
#plt.contourf(Xim ,Yim , np.reshape(Zi0N, Xim.shape), 60)
#plt.colorbar()
#
#plt.plot(Xcoord[:,0],Xcoord[:,1], 'ow')
##plt.plot(xr, yr, 'or', linewidth = 10)
##plt.plot(xy[i][0],xy[i][1], 'or', linewidth = 10)
##plt.plot(Xcoord[id_seism ,0],Xcoord[id_seism ,1], 'or', linewidth = 10)
#plt.title('Nearest')

