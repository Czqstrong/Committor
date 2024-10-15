#process data
#Use for padding

import numpy as np

L       = 8
d       = L**2;
alpha   = 10;
seed    = 6;
verbose = 1;
n_e     = 6;
BATCH_SIZE = 200;
w_size1  = L;
w_size2  = L;

dataset = np.loadtxt("homosample_n=8.txt", delimiter=",", usecols=range(d+1))
X = dataset[:,1:d+1]
Y = dataset[:,0]

X_new = np.zeros((Y.shape[0], 2*L-1, 2*L-1, 1));
count = 0;

for i in range(0,Y.shape[0]):

    tmpx  = X[i,0:L**2];
    tmpx   = np.reshape(tmpx,(L,L));

    tmpx = np.concatenate((tmpx,tmpx[:,0:w_size1-1]),axis=1)
    tmpx = np.concatenate((tmpx,tmpx[0:w_size2-1,:]),axis=0)


    X_new[count,:,:,0] = tmpx
    count+=1;

X = X_new;


np.save('X_n=8.npy',X)
np.save('Y_n=8.npy',Y)
