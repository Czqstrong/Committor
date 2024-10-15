#Simple one hidden layer feed-forward
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import Convolution2D
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Activation
from keras.models import Model
from keras.layers import BatchNormalization
#import h5py
from keras import backend as K
import numpy as np

global alpha
global d
global w_size
L       = 8

xfile   = 'X_n=8.npy'
yfile   = 'Y_n=8.npy'
d       = L**2;
alpha   = 16;
seed    = 6;
verbose = 1;
n_e     = 10;
BATCH_SIZE = 300;
p_test   = 0.5;
w_size   = L;


#lambda layer
def pooling(x):
    return K.sum(x,axis=(K.ndim(x)-2,K.ndim(x)-1))


X = np.load(xfile)
Y = np.load(yfile)

n = Y.shape[0];
Xtest  = X[0:n//2-1,:,:,:];
Ytest  = Y[0:n//2-1];
Xtrain = X[n//2:n,:,:,:];
Ytrain = Y[n//2:n];

model = Sequential();
model.add(Convolution2D(alpha,(w_size,w_size),activation='relu',input_shape=(L+w_size-1,L+w_size-1,1),data_format='channels_last'))
print(model.output)
model.add(Flatten())

model.add(Dense(1,activation='linear'))
print(model.output)


# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')
model.optimizer.schedule_decay = (0.004)
model.optimizer.lr = (0.0003)


model.fit(Xtrain,Ytrain, batch_size=BATCH_SIZE, nb_epoch=2000, verbose=1)
Yhat = model.predict(Xtest);


err  = np.ndarray.flatten(Yhat)-np.ndarray.flatten(Ytest);

err  = np.linalg.norm(err)/np.linalg.norm(Ytest)
print(err)


#scores = model.evaluate(Xtest, Ytest)
#print(scores)

#W = model.get_weights();
#model.save_weights("2dweightn=16_1.h5")

#print("\n%s: %.2f%%" % (model.metrics_names[0], scores[1]))
