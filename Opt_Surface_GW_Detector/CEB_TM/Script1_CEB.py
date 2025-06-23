from gwpy.timeseries import TimeSeries
import pickle

import numpy as np
import sys
import time 
import math as mt

from pyKriging.krige import kriging  
from numpy import matlib
from scipy import signal
from scipy import fftpack
from functools import partial
from multiprocessing import Pool


Path = '/data/rawdata/rawback/'
#Path = ''
listofdata0 = []

if (len(sys.argv) > 0):
        for i in sys.argv[1:]:
                listofdata0.append(i)

listofdata = [Path + i for i in listofdata0]
list_files = ['CEB_LEV1.txt', 'CEB_LEV2.txt']
#list_files = ['CEB_LEV2.txt']
print (len(sys.argv)-1)
calib_int = 5/(2**23*77.3)
Channel = []
x = []
y = []

t0 = time.time()
for lf in list_files:
        with open(lf) as file:
                c = 0
                for line in file:
                        if (c > 0):
                                row = line.split('	')
                                #print(row)
                                Channel.append(row[1])
                                x.append(float(row[6]))
                                y.append(float(row[7]))
                        c = c+1 
           
fs = 250 #data are sampled at 500 Hz and I want to resample them at 250 Hz

samples_int = np.zeros((len(listofdata0)*100*fs, len(Channel)))


##Removing the bad_sensor in this way produces some error in the kriging process 
#bad_sensor = 'V1:NN_CEB_ACC_Y_LEV2_35'
#if (bad_sensor in Channel):
#	indx_bad_sensor = Channel.index(bad_sensor)
#	Channel.remove(bad_sensor)
#	x.pop(indx_bad_sensor)
#	y.pop(indx_bad_sensor)

X = np.zeros((len(x),2))
X[:,0] = x
X[:,1] = y

#check:
if (not(len(x)==len(y)==len(Channel))):
	print('error')
	sys.exit()

c = 0
for ch in Channel:
     tmp = TimeSeries.read(listofdata, ch)
     tmp = tmp.detrend(detrend = 'linear')
     tmp = tmp.resample(fs)
     samples_int[:,c] = tmp.value*calib_int
     print(Channel[c])
     c += 1

t1 = time.time()
print('Time elapsed: ', t1-t0)

nfft = 1024 

L = nfft
D = int(L/2) # overlapping   
Lt = samples_int.shape[0]
n_loop = int(Lt/(L-D)) 
if (Lt-(n_loop*(L-D)) < L-D):
	n_loop = n_loop -1	

freq_fft = np.fft.fftfreq(L, 1/fs)

krigListRe = []
krigListIm = []
Hann_win = np.matlib.repmat(signal.windows.hann(L, sym=False),samples_int.shape[1],1).transpose()
U = np.average(Hann_win[:,0]**2, axis = 0)



barLength = 100

def foopiee(iii):
	global Hann_win
	global samples_int
	seg = Hann_win*samples_int[iii*(L-D):iii*(L-D)+L,:]
	fft_int = fftpack.fft(seg, n=None, axis=0, overwrite_x=False)
	
	freq_id = mt.ceil(15/fs*nfft) # 15 Hz with L=10204 and fs = 250
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

if (__name__ == '__main__'):
	
	#a = foopiee(0)
	#print(a)
	#print(X)
	#[print(a[0].predict_var([X[i,0], X[i,1]])) for i in range(0, X.shape[0])]
	#x1 = np.random.randn(len(x))+X[:,0]
	#y1 = np.random.randn(len(y))+X[:,1]
	#[print(a[0].predict([x1[i], y1[i]])) for i in range(0, X.shape[0])]
	#[print(a[0].predict_var([x1[i], y1[i]])) for i in range(0, X.shape[0])]
	#a[0].plot()
	
	t0 = time.time()
	print(n_loop)


	Poo = Pool(6)
	KL = Poo.map(foopiee, list(range(0,n_loop)))
	Poo.close()
	Poo.join()
	t1 = time.time()
	print('elapsed time:  ',t1-t0)

	#%% SAVE
	
	filehandler = open('KL_scipy_15Hz.obj', 'wb') 
	pickle.dump(KL, filehandler)
	filehandler.close()



