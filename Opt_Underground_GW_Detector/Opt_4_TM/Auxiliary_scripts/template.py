import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')
SNR = 15
f = 10


e2 = 64.12*np.array([0.5,np.sqrt(3)/2,0])
e1 = 64.12*np.array([1,0,0])
e3 = 536.35*np.array([0.5,np.sqrt(3)/2,0])
e4 = 536.35*np.array([1,0,0])

plt.figure(1, figsize=(12,9))
ax.scatter(e1[0],e1[1],e1[2],c='r', marker='o')
ax.scatter(e2[0],e2[1],e2[2],c='r', marker='o')
ax.scatter(e3[0],e3[1],e3[2],c='r', marker='o')
ax.scatter(e4[0],e4[1],e4[2],c='r', marker='o')
plt.plot([0,e4[0]], [0,e4[1]], '--', c='k')
plt.plot([0, e3[0]], [0, e3[1]], '--', c = 'k')
#plt.show()
plt.xlabel('X')
plt.ylabel('Y')
