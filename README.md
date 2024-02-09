# MatchingPursuit
Matching Pursuit method algorithm.

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm,pinv

eps=10**-4
iterMax=10000

D1=np.array([[np.sqrt(2)/2,np.sqrt(3)/3,np.sqrt(6)/3,2/3,-1/3],
[-np.sqrt(2)/2,-np.sqrt(3)/3,-np.sqrt(6)/6,2/3,-2/3],
[0,-np.sqrt(3)/3,np.sqrt(6)/6,1/3,2/3]],float)

#signal 1

x1=np.array([4/3-np.sqrt(2)/2,4/3+np.sqrt(2)/2,2/3],float)

D2=np.array([[1,1,2,5,0,0,3,-2,1,2,2,2],
[ 0,-1,-1,1,0,0,5,0,2,2,7,-1],
[1,1,1,5,1,2,2,1,1,1,1,5],
[1,5,2,2,5,0,-4,5,1,5,0,0],
[0,2,2,1,1,0,0,0,0,4,-1,-2],
[-1,2,2,2,-2,-3,-4,1,1,1,1,0]])

#Signal 2

x2=np.array([-10,-10,1,21,0,9])

D3=np.array([[1,1,2,5,0,0,3,-2,1,2,2,2,5,1,3,1,-1,2,9,5,5,1,1,5],
[0,-1,4,2,-1,1,0,0,5,0,2,2,7,-12,2,5,5,2,7,4,-9,-2,1,2],
[1,3,1,1,5,1,2,2,1,1,1,1,5,0,-1,1,0,1,2,1,1,2,5,5],
[0,1,5,1,5,2,2,-2,5,0,-4,5,1,5,0,0,-1,-4,-8,2,2,-1,1,0],
[0,-1,2,3,2,2,3,1,1,0,0,0,0,4,-1,-2,0,7,4,3,4,-1,1,0],
[-1,8,6,3,2,2,2,4,-2,-3,-4,1,1,1,1,0,-2,-3,4,1,1,-1,1,0]])
#Signal 2
x3=np.array([-10,-10,10,20,15,10])

D=D3
x=x3
[n,k]=np.shape(D)
alpha=np.zeros(k)
R=x
it=0
index=[]
while norm(R)>eps and it<iterMax:
	ps=np.zeros(k)
	for i in range(k):
		ps[i]=np.abs(np.dot(D[:,i].T,R))/norm(D[:,i])
	m=np.argmax(ps)
	index.append(m)
	A=D[:,index]
	alpha[index]=np.dot(pinv(A),x)
	R=x-np.dot(D,alpha)
	it+=1
print('alpha=', alpha)
print('norm residu=', norm(R))
print('nb iter=', it)
