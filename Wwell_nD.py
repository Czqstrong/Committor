# Double potential well: Uniform sampling on x_1


import scipy.integrate as integrate
import numpy as np

np.random.seed(24)  # for reproducibility
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, BatchNormalization, Add, Multiply
import keras.optimizers as optimizers
from keras import regularizers
from keras.engine.topology import Layer
import matplotlib.pyplot as plt
import keras.backend as K


# 2. Define gradient norm

class GradNorm(Layer):
    def __init__(self, **kwargs):
        super(GradNorm, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(GradNorm, self).build(input_shapes)

    def call(self, inputs):
        target, wrt = inputs
        grads = K.gradients(target, wrt)
        assert len(grads) == 1
        grad = grads[0]
        return K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)


class BC(Layer):
    def __init__(self, bv, **kwargs):
        self.bv = bv
        super(BC, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(BC, self).build(input_shapes)

    def call(self, inputs):
        q = inputs
        bv = self.bv
        return K.sum(K.batch_flatten(K.square((q - bv))), axis=1, keepdims=True)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[0], 1)


def cost(ytrue, ypred):
    return K.sum(K.sum(ypred))


def potential(x, b):
    V = (x[0, 0] ** 2 - 1 ** 2) ** 2 + b * np.sum(x[0, 1:np.size(x)] ** 2)

    F = -np.hstack((4 * (x[0, 0] ** 2 - 1) * x[0, 0], 2 * b * x[0, 1:np.size(x)]))
    return V, F


def sampling(N, N_bound, h, p, margin, b, T, r):
    ct = 1
    tmp = np.zeros((1, 1))
    Xmid = tmp
    while ct < N:
        tmp = 2 * (-0.5 + np.random.rand(1, 1))
        Xmid = np.vstack((Xmid, tmp))
        ct = ct + 1
        print(ct)

    print(Xmid.shape)
    Xmid = np.hstack((Xmid, 1. / np.sqrt(T / 2. / b) * np.random.randn(N, p - 1)))

    idx = np.random.randint(N, size=(N_bound))

    X_A = Xmid[idx, :]
    X_B = Xmid[idx, :]
    X_A[:, 0] = -r
    X_B[:, 0] = r

    yA = np.zeros((2 * N_bound + N, 1))
    yB = np.zeros((2 * N_bound + N, 1))
    yin = np.zeros((2 * N_bound + N, 1))
    yA[0:N_bound, :] = 1
    yB[N_bound:2 * N_bound, :] = 1
    yin[2 * N_bound:2 * N_bound + N, 0] = np.exp(-1.0 / T * (Xmid[:, 0] ** 2 - 1 ** 2) ** 2) * 100
    X = np.vstack((X_A, X_B, Xmid))
    Y = [yin, yA, yB]

    return X, Y


# 3. Create samples
N = 20 * 10 ** 3
N_bound = 1000
rho = 0.1 * 10 ** 2
p = 3
r = 1
h = 0.007
T = 0.05
Nepoch = 5000
split = 1
b = 0.3
margin = r

Xin, Yin = sampling(N, N_bound, h, p, margin, b, T / split, r)

plt.figure(0)
plt.hist2d(Xin[:, 0], Xin[:, 1], 100)

plt.xlabel(r'$x_1$', fontsize=16)
plt.ylabel(r'$x_2$', fontsize=16)
plt.show()

# Define committor function q
L1 = Input(shape=(p,))
L2 = Dense(12, activation='tanh', kernel_regularizer=regularizers.l2(0.00))(L1)
L2 = Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.00))(L2)
q = Model(inputs=L1, outputs=L2)

# Build training model
M1 = [Input(shape=(p,)), Input(shape=(1,)), Input(shape=(1,)), Input(shape=(1,))]
M2 = q(M1[0])

norm = GradNorm(name='Gnorm')([M2, M1[0]])
norm = Multiply()([norm, M1[1]])

B_A = BC(bv=0)(M2)
B_A = Multiply()([B_A, M1[2]])

B_B = BC(bv=1)(M2)
B_B = Multiply()([B_B, M1[3]])

added = Add()([norm, B_A, B_B])
model = Model(inputs=M1, outputs=added)

Nadam = optimizers.adam(lr=0.002, beta_1=0.9, beta_2=0.999, decay=0.0002)
model.compile(loss=cost, optimizer=Nadam)

Ytrain = np.zeros((np.size(Xin, 0), 1))
Xtrain = [Xin, Yin[0], Yin[1] * rho, Yin[2] * rho]

model.fit(Xtrain, Ytrain,
          batch_size=3000, nb_epoch=5000, verbose=1)


# ========== Compute q-error ==============%
def Wq(x, beta, r):
    const1 = integrate.quad(lambda x: np.exp(beta * (x ** 4 - 2 * x ** 2)), -r, r)
    const1 = 1 / const1[0]

    q = const1 * integrate.quad(lambda x: np.exp(beta * (x ** 4 - 2 * x ** 2)), -r, x[0])[0]
    for i in range(1, x.shape[0]):
        tmp = const1 * integrate.quad(lambda x: np.exp(beta * (x ** 4 - 2 * x ** 2)), -r, x[i])[0]
        q = np.vstack((q, tmp))

    return q


def Wq_grad(beta, r):
    const1 = integrate.quad(lambda x: np.exp(beta * (x ** 4 - 2 * x ** 2)), -r, r)
    const1 = 1 / const1[0]

    g = integrate.quad(
        lambda x: np.exp(-beta * (x ** 2 - 1) ** 2) * const1 ** 2 * np.exp(beta * (x ** 4 - 2 * x ** 2)) ** 2, -r, r)[0]
    normalization = integrate.quad(lambda x: np.exp(-beta * (x ** 2 - 1) ** 2), -r, r)[0]

    return g / normalization


# ========== testing data ===========#

Xline = np.linspace(-r, r, 50)
Yline = np.zeros((p - 1, 50))
Xdata = np.vstack((Xline, Yline))
Xdata = Xdata.T

z1 = q.predict(Xdata)

Yline[0, :] = 0.8
Xdata = np.vstack((Xline, Yline))
Xdata = Xdata.T
z2 = q.predict(Xdata)

Yline[1, :] = 0.5
Xdata = np.vstack((Xline, Yline))
Xdata = Xdata.T
z3 = q.predict(Xdata)

pred = Wq(Xline, 1 / T, r)

fig, ax = plt.subplots()
ax.plot(Xline, z1, 'b-', label='NN')
ax.plot(Xline, pred, 'r-', label='Committor function')
legend = ax.legend(loc='upper left', shadow=True)

plt.xlabel(r'$x_1$', fontsize=20)
plt.ylabel('Committor function', fontsize=16)

plt.show()

# =========== Compute q-error ============#
Ntest = 1 * 10 ** 5
Xtest, Ytest = sampling(Ntest, 0, h, p, margin, b, T, r)
qtest = q.predict(Xtest)
rtest = Xtest[:, 0]
qpred = Wq(rtest, 1 / T, r)

err = np.linalg.norm(qtest - qpred) / np.linalg.norm(qpred)

# =========== Compute energy norm error ==========#
L1 = Input(shape=(p,))
L2 = q(L1)
L3 = GradNorm()([L2, L1])
gnorm = Model(inputs=L1, outputs=L3)

qgrad = np.sum(gnorm.predict(Xtest))
qgrad = qgrad / Ntest

enorm = Wq_grad(1 / T, r)
err_norm = abs(qgrad - enorm) / enorm
