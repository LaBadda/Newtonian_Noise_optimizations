import numpy as np
import pandas as pd
import sys
import time 
import math as mt


from pyKriging.krige import kriging  
from scipy import signal
from scipy import fftpack
from functools import partial
from multiprocessing import Pool
#from functools import partial
#from scipy import stats 
from itertools import combinations	

#from matplotlib import pyplot as plt


# %% Load data .csv

t0 = time.time()
df = pd.read_csv('/storage/gpfs_small_files/VIRGO/users/fbadaracco/input/180205000000.badCRC_0.out.csv',sep = ';', header=None, index_col=0, compression = None)
#df = pd.read_csv('C:/Users/Francesca/Documents/VirgoWEB/180205000000.badCRC_0.out.csv',sep = ';', header=None, index_col=0)
df = df.dropna(axis=1)
#d = df.values

## Transform the int64 index in DataIndex for the resampling 
#df.index = pd.to_datetime(df.index + np.int64(20e+15), format = '%Y%m%d%H%M%S%f')
t1 = time.time()
print('Time elapsed: ', t1-t0)

t = df.index
samples_int = df.values
#del df

ns = np.int(np.floor(samples_int.shape[0]/2))

calib_int = 5/(2**23*77.3)  #calibration factor of indoor sensors
samples_int = signal.detrend(samples_int*calib_int, axis=0, type='constant')
dec_factor = 2
samples_int = signal.decimate(samples_int, dec_factor, axis = 0, zero_phase=True)

fs = 250
nfft = 1024
	
#%% Kriging of Real part of Fourier Transform
	
coords = np.array([[-2996.1855, 3.3109, -3.4011],[-2998.6876, 7.1706, -3.4071],[-2999.3596, 3.2421, -3.4008],[-3003.9149, 7.8757, -3.4227],[-3003.0811, 3.2749, -3.4150],[-3005.9728, 7.7190, -3.4288],[-3006.2434, 3.3279, -3.4237],[-3010.3192, 7.9888, -3.4437],[-3012.1375, 3.3488, -3.4299],[-3014.7164, 7.5541, -3.4490],[-3017.8701, 4.2944, -3.4613],[-3014.9770, -0.0874, -3.4426],[-3017.9104, -4.0994, -3.4595],[-3015.1017, -7.3702, -3.4573],[-3013.7171, -3.1819, -3.4287],[-3011.1778, -6.0311, -3.4400],[-3009.9611, -3.1498, -3.4302],[-3005.5196, -6.1313, -3.4285],[-3005.9568, -3.1540, -3.4234],[-3003.6825, -6.6868, -3.4195],[-3000.1694, -3.4201, -3.4149],[-2999.4935, -5.4172, -3.4146],[-2996.6250, -3.3377, -3.4085],[-2999.3105, 2.7366, -3.3156],[-3002.9757, 2.6436, -3.3247],[-3006.8761, 2.7944, -3.3304],[-3010.9353, 2.8204, -3.3416],[-3013.8210, 2.3477, -3.3491],[-3013.6882, -2.6371, -3.3435],[-3011.2803, -2.5983, -3.3446],[-3006.4946, -2.6779, -3.3386],[-3002.6733, -2.5689, -3.3296],[-2999.6753, -2.5736, -3.3223],[-2999.8896, 0.0321, -3.3182],[-3009.4183, 0.0789, -3.3497],[-3013.7110, -0.0613, -3.3462],[-3012.2423, -1.1159, -6.8445],[-3008.4083, 1.4434, -6.8571]])
X = coords[:,0:2]
X = np.delete(X,  [34], axis=0)

x0 = -3005.5847
y0 = 0.0312
h = np.average(coords[0:21,2]) - (-2.2233)

#centering around test mass
X[:,0] = X[:,0] - x0
X[:,1] = X[:,1] - y0

ii_ref = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,35,36,37]
t0 = time.time()
samples_int2 = samples_int[:,ii_ref]

L = 1024
D = int(L/2) # overlapping   
Lt = samples_int.shape[0]
n_loop = int(Lt/(L-D)) 
if (Lt-(n_loop*(L-D)) < L-D):
	n_loop = n_loop -1	

freq_fft = np.fft.fftfreq(L, 1/fs)

krigListRe = []
krigListIm = []
Hann_win = np.matlib.repmat(signal.windows.hann(L, sym=False),len(ii_ref),1).transpose()
#Hann_win = signal.get_window('hann',L) #same
U = np.average(Hann_win[:,0]**2, axis = 0)
#	U = np.average(Hann_win[:,0], axis = 0)**2



barLength = 100

#	KL = [None]*n_loop
#	for iii in range(0, n_loop):		
def foopiee(iii):
	global Hann_win
	global samples_int2
	seg = Hann_win*samples_int2[iii*(L-D):iii*(L-D)+L,:]
	fft_int = fftpack.fft(seg, n=None, axis=0, overwrite_x=False)
	
	freq_id = mt.ceil(10/fs*nfft) # 15 Hz with L=10204 and fs = 250
	
	a1 = (np.real(fft_int[freq_id,:]))
	a2 = (np.imag(fft_int[freq_id,:]))
	
	# Now that we have our initial data, we can create an instance of a Kriging model
	kRe = kriging(X, a1, testfunction=None, name='simple')
	kRe.train()
	kIm = kriging(X, a2, testfunction=None, name='simple')  
	kIm.train()

#	progress = (iii+1)/n_loop
#	block = round(barLength*progress)
#	print("\rPercent: [{0}] {1:.2f}".format( "#"*block + "-"*(barLength-block), progress*100),end="", file = sys.stdout)
	return [kRe,kIm]
#	KL[iii] = [kRe, kIm]

if (__name__ == '__main__'):
	t0 = time.time()
	Poo = Pool(4)
	KL = Poo.map(foopiee, list(range(0,n_loop)))
	Poo.close()
	Poo.join()
	t1 = time.time()
	print('elapsed time:  ',t1-t0)

	#%% SAVE
	import pickle 
	
	filehandler = open('KL_scipy_mine.obj', 'wb') 
	pickle.dump(KL, filehandler)
	filehandler.close()


