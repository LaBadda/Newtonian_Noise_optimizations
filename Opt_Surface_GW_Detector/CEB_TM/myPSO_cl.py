import numpy as np
# from itertools import combinations	
from scipy.integrate import simps
from scipy import linalg
import sys


class myPSO():
	def __init__(self, xm, xM, ym, yM, InterPolIm, InterPolRe):
		self.xm = xm
		self.xM = xM
		self.ym = ym
		self.yM = yM
		
		if ((type(InterPolRe) is list) & (type(InterPolIm) is list)):
			self.InterPolRe_l = InterPolRe
			self.InterPolIm_l = InterPolIm
			self.l = len(self.InterPolIm_l)
		else:
			self.InterPolRe = InterPolRe
			self.InterPolIm = InterPolIm
			

	def Css_matrix(self,Ip,ns):
		
		"""
		Ip are the positions of the ns seismometers coming from the optimization process
		
		"""
		Css_vecIm = self.InterPolIm.__call__(Ip)
		Css_vecRe = self.InterPolRe.__call__(Ip)
	
		Css = Css_vecRe.reshape(ns,ns) + 1j*Css_vecIm.reshape(ns,ns)
		
		if (np.any(np.isnan(Css))):
			print('Warning. NaN in Css : ', Css , file = sys.stderr)	
	
		Diag = np.diag(Css)
		Nfact = np.sqrt(np.tensordot(Diag,Diag, axes = 0))	
		Css = Css/Nfact
		
				
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
		
	def check_rect(self,x,y,x_left, x_right, y_low, y_up):
		if ((x > x_left) & (x < x_right) & (y > y_low) & (y < y_up)):
			return True
		else:
			return False
		
	def Csn_vector (self, x,y,xt,yt, xs, ys,dim,TM = 'x'): 
	
		"""xs and ys are the points defining the grid over which perform the integral"""
	
		ns = len(x)
		Csn = np.zeros(ns, dtype = complex)
		for uu in range(0,ns):
			xsm, ysm = np.meshgrid(xs,ys)

			x0 = np.ones(xsm.shape)*xt
			y0 = np.ones(ysm.shape)*yt

			Ip = np.stack((xsm.flatten(), ysm.flatten(),x[uu]*np.ones(xsm.shape[0]*xsm.shape[1]), y[uu]*np.ones(xsm.shape[0]*xsm.shape[1])), axis = 1)
			Csn_v = self.InterPolRe.__call__(Ip) + 1j*self.InterPolIm.__call__(Ip)
			
			if (self.check_rect(x[uu],y[uu], -1.75, 1.75, 1.75, 8.95) |
				self.check_rect(x[uu],y[uu], -8.15, 12.90, -1.75, 1.75) | 
				self.check_rect(x[uu],y[uu], -1.75, 1.75, -12.75, -1.75) | 
				self.check_rect(x[uu],y[uu], 1.75, 4.6, -8.5, -1.75) |  #WROOONGG!!!!!!!!!!
				self.check_rect(x[uu],y[uu], 4.6, 7.75, -13.35, -1.75)):
				h = 4.95
			else:
				h = 1.1
			if (TM=='x'):
				Csn_s = -np.reshape(Csn_v,dim)*((x0 - xsm)/((h**2 + (x0 - xsm)**2 + (y0 - ysm)**2)**(3/2)))
			elif (TM=='y'):
				Csn_s = np.reshape(Csn_v,dim)*((y0 - ysm)/((h**2 + (x0 - xsm)**2 + (y0 - ysm)**2)**(3/2)))
			else:
				print('wrong TM')
				
			#	Integration along xs:
			Csnx = simps(Csn_s, xs, axis = 1)
			Csn[uu] = simps(Csnx, ys)
	
		if (np.any(np.isnan(Csn))):
			print('Warning. NaN in Csn: ', Csn , file = sys.stderr)	
		
		return Csn
	
	def Res(self, state, xs, ys,xx,yx, xy, yy, dim, ns):
		'''xx,yx: x arm test mass coordinates wrt the BS'''
		'''xy,yy: y arm test mass coordinates wrt the BS'''
	
		#print(state.shape)
		if len(state.shape) == 1:
			state = np.expand_dims(state, 0)

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
	
			if (((min(x) < self.xm) | (max(x) > self.xM) | (min(y) < self.ym) | (max(y) > self.yM)) == 1):
				raise ValueError('Boundaries violated')
	
	#		snr = 1e-20
			Css = self.Css_matrix(Ip,ns)# + np.diag(np.ones(ns)*snr)
	
			# Csn:
			Csn = self.Csn_vector(x,y, xx,yx, xs, ys, dim, 'x') - self.Csn_vector(x,y, xy,yy, xs, ys,dim,'y')
			if((np.sum(np.isnan(Csn))) & (np.sum(np.isnan(Css.flatten()))) >= 1):
				print("c'è un NaN(o)")
			
	
			residual = -np.dot(Csn.conjugate(),np.dot(Css,Csn))
			Res_Vec[tt] = np.real(residual)	
		
		return Res_Vec





	def Csnxy(self, state, xs, ys,xx,yx, xy, yy, dim, ns):
		'''xx,yx: x arm test mass coordinates wrt the BS'''
		'''xy,yy: y arm test mass coordinates wrt the BS'''
	
		#print(state.shape)
		if len(state.shape) == 1:
			state = np.expand_dims(state, 0)

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
	
			if (((min(x) < self.xm) | (max(x) > self.xM) | (min(y) < self.ym) | (max(y) > self.yM)) == 1):
				raise ValueError('Boundaries violated')
	
	#		snr = 1e-20
			Css = self.Css_matrix(Ip,ns)# + np.diag(np.ones(ns)*snr)
	
			# Csn:
			Csn = self.Csn_vector(x,y, xx,yx, xs, ys, dim, 'x')*self.Csn_vector(x,y, xy,yy, xs, ys,dim,'y').conjugate()
# 			Csnx = np.real(self.Csn_vector(x,y, xx,yx, xs, ys, dim, 'x'))#*self.Csn_vector(x,y, xx,yx, xs, ys, dim, 'x').conjugate()
# 			Csny = np.real(self.Csn_vector(x,y, xy,yy, xs, ys, dim, 'y'))#*self.Csn_vector(x,y, xy,yy, xs, ys, dim, 'y').conjugate()
			
	
# 			residual = -np.dot(Csn.conjugate(),np.dot(Css,Csn))
			Res_Vec[tt] = np.real((Csn)*Css)	
		
		return Res_Vec





	def Res1fix(self, state, xs, ys,xx,yx, xy, yy, xf,yf, dim, ns):
		'''xx,yx: x arm test mass coordinates wrt the BS'''
		'''xy,yy: y arm test mass coordinates wrt the BS'''
	
		#print(state.shape)
		if len(state.shape) == 1:
			state = np.expand_dims(state, 0)

		state[:,2] = np.ones(state.shape[0])*xf
		state[:,3] = np.ones(state.shape[0])*yf
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
	
			if (((min(x) < self.xm) | (max(x) > self.xM) | (min(y) < self.ym) | (max(y) > self.yM)) == 1):
				raise ValueError('Boundaries violated')
	
	#		snr = 1e-20
			Css = self.Css_matrix(Ip,ns)# + np.diag(np.ones(ns)*snr)
	
			# Csn:
			Csn = self.Csn_vector(x,y, xx,yx, xs, ys, dim, 'x') - self.Csn_vector(x,y, xy,yy, xs, ys,dim,'y')
			if((np.sum(np.isnan(Csn))) & (np.sum(np.isnan(Css.flatten()))) >= 1):
				print("c'è un NaN(o)")
			
	
			residual = -np.dot(Csn.conjugate(),np.dot(Css,Csn))
			Res_Vec[tt] = np.real(residual)	
		
		return Res_Vec











	def BroadBandRes(self, state, xs, ys, dim, ns):
		
		n_part = state.shape[0]
		Res_Vec = np.zeros(n_part)
		
		for tt in range(0,n_part):
			R = [None]*self.l
			for i in range(0, self.l):
				self.InterPolIm = self.InterPolIm_l[i]
				self.InterPolRe = self.InterPolRe_l[i]
				R[i] = self.Res( state[tt,:], xs, ys, dim, ns)[0]
			Res_Vec[tt] = np.max(R)
		return Res_Vec
