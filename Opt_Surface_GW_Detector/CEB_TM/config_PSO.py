import numpy as np
import pickle
import scipy.interpolate as In

Xc = np.zeros((49,2))

#without sensor numb 35 lev 2
Xc[:,0] = np.array([ -1.8,  -7.8,  -7.8,  -1.8,  -1.8,   0.7,   0.7,   8.1,  -1.8, 0.7,   1.3,   6.8,   6.8,   3.9,   3.8,   1.5,   2.1,   2.1, -0.7,  -2.1,  -2.1,  -0.7,  -0.7,   0.6,  -5.7,  -8.5, -11.4, -7.7,  -6.3,  -3.5,  -2.1,  -3.4,   5.4,  -5.2,  -8.9, -10. , 5.4,   5.4,  -0.8,  -3.6,  -6.2,  -6.8, -12.2, -17.1, -19. , -18.6, -13.5,  14.8,  14.8])
Xc[:,1] = np.array([ -1.8 ,  -1.8 ,   0.6 ,   1.  ,   7.4 ,   7.4 ,   0.8 ,   0.8 , -10.1 , -10.1 ,  -7.9 ,  -8.9 ,  -2.  ,  -2.4 ,  -5.3 ,  -4.9 , 6.65,   4.15,   3.65,   5.05,   7.25,   8.95,  11.85,   8.95, -2.  ,  -0.6 ,  -0.9 ,   1.25,   2.9 ,   0.9 ,  -0.1 ,  -1.4 , 9.2 ,   5.2 ,   4.8 ,   6.1 ,  13.3 ,  17.1 ,  17.9 ,  13.8 , -9.3 ,  -5.2 ,  -3.6 ,  -3.6 ,  -0.9 ,   3.7 ,   3.7 ,  -1.8 , 5.7 ])

xm = Xc[:,0].min() 
xM = Xc[:,0].max()
ym = Xc[:,1].min()
yM = Xc[:,1].max()

Lx = xM - xm
Ly = yM - ym


P = './'


pickle_in = open(P+"CSSRe.obj","rb")
CssRe = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open(P+"CSSIm.obj","rb")
CssIm = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open(P+"x1.obj","rb")
x1 = pickle.load(pickle_in)
pickle_in.close()
	
pickle_in = open(P+"y1.obj","rb")
y1 = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open(P+"x2.obj","rb")
x2 = pickle.load(pickle_in)
pickle_in.close()
	
pickle_in = open(P+"y2.obj","rb")
y2 = pickle.load(pickle_in)
pickle_in.close()

Xcomb = np.stack((x1,y1,x2,y2), axis = 1)

X = np.unique(x1)
Y = np.unique(y1)
Xm1, Ym1, Xm2, Ym2 = np.meshgrid(X,Y,X,Y, indexing='ij')
CSSregridRe = CssRe.reshape(Xm1.shape)
CSSregridIm = CssIm.reshape(Xm1.shape)

InterPolRe = In.RegularGridInterpolator((X,Y,X,Y), CSSregridRe, method='linear')
InterPolIm = In.RegularGridInterpolator((X,Y,X,Y), CSSregridIm, method='linear')
