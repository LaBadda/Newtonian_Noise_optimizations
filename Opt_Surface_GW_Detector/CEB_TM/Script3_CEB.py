import pickle 
import numpy as np


N = 30**4
CSS = np.zeros(N,dtype = complex)
x1 = np.zeros(N)
y1 = np.zeros(N)
x2 = np.zeros(N)
y2 = np.zeros(N)

P = 'C:/Users/badaracco/OneDrive - UCL/Documents/Projects/CEB_Opt/Res2/'
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

Xcomb = np.stack((x1,y1,x2,y2), axis = 1)
CssIm = np.imag(CSS)
CssRe = np.real(CSS)




filehandler = open('CSSIm.obj', 'wb') 
pickle.dump(CssIm, filehandler)
filehandler.close()

filehandler = open('CSSRe.obj', 'wb') 
pickle.dump(CssRe, filehandler)
filehandler.close()

filehandler = open('x1.obj', 'wb') 
pickle.dump(x1, filehandler)
filehandler.close()

filehandler = open('y1.obj', 'wb') 
pickle.dump(y1, filehandler)
filehandler.close()

filehandler = open('x2.obj', 'wb') 
pickle.dump(x2, filehandler)
filehandler.close()

filehandler = open('y2.obj', 'wb') 
pickle.dump(y2, filehandler)
filehandler.close()