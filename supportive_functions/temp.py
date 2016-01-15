import numpy as np
from numpy.linalg import inv

##coordination system definition
x0_v,y0_v,z0_v=np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.])
#anonymous function f1 calculating transforming matrix with the basis vector expressions,x1y1z1 is the original basis vector
#x2y2z2 are basis of new coor defined in the original frame,new=T.orig
f1=lambda x1,y1,z1,x2,y2,z2:np.array([[np.dot(x2,x1),np.dot(x2,y1),np.dot(x2,z1)],\
                                      [np.dot(y2,x1),np.dot(y2,y1),np.dot(y2,z1)],\
                                      [np.dot(z2,x1),np.dot(z2,y1),np.dot(z2,z1)]])
BASIS=np.array([5.1988, 9.0266, 20.1058])
BASIS_SET=[[1,0,0],[0,1,0],[0.10126,0,1.0051136]]
T=f1(x0_v,y0_v,z0_v,*BASIS_SET)
T_INV=inv(T)

#print T
print np.dot(np.transpose(T_INV),[0,0,1])
