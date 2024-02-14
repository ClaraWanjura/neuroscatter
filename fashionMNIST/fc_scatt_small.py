# -*- coding: utf-8 -*-
"""
Neuromorphic Setup (Nonlinear Transformation from Linear Wave Scattering)

Code for training a neuromorphic device implementing nonlinear
information processing via the parameter dependence of the
scattering matrix.

Attributes implemented here (see manuscript):
- Transmission Setup
- Fully Connected Structure
- Data set: Fashion MNIST

Requirements: jax, optax, tensorflow [only for data loading]

Train the parameters in a scattering matrix using jax.

NOTE:
Directory 'outputs' needs to exist in the current directory!

C. Wanjura and F. Marquardt
2023/24, MIT License
"""

# Parameters

scale_down_images=True # if True, really use 14x14 instead of 28x28
input_scale=5 # scaling of input values physically injected into detunings of optical modes

random_scale0=1 # global scaling of random initialization of couplings (fully connected 0-1)
random_scale1=1 # global scaling of random initialization of couplings (fully connected 1-2)

random_layer0_detuning_scale = 5 # random init of layer 0 detunings
random_detuning_scale = 0.05 # random init scale of detunings

N1=784//4 # hidden layer neuron modes

batch_size=20 # number of samples per batch
train_batches=300_000 # how many training batches to do (note 1 epoch = 60_000 samples)
max_learning_rate=0.01 # for other details, look below for schedule
end_learning_rate=0.0

import optax

optimizersched = optax.chain(
    optax.amsgrad(learning_rate=max_learning_rate),
    optax.clip(0.001)
)

DIRNAME="./outputs/" # intermediate and final results go inside here



# Import necessary libraries

import numpy as np
from numpy import loadtxt

import jax.numpy as jnp
import jax
from jax import jit, vmap, grad, value_and_grad

import tensorflow as tf
import keras

# Do not waste GPU memory on tensorflow:
tf.config.experimental.set_visible_devices([], "GPU")

import matplotlib.pyplot as plt
import time


start_time=time.time()

if scale_down_images:
    scale_image_factor=4
else:
    scale_image_factor=1

"""# Definition of Network"""

def invSuscept(gamma, omega):
    return jnp.diag(gamma/2 - 1j*omega)

def suscept(gamma, omega):
    return jnp.diag(1/(gamma/2 - 1j*omega))


## initialise gamma and omega vectors and JJ with random values
## Nj is the size of the jth layer
def randomVector(Nj):
    return np.random.uniform(size=(Nj,))

def randomJJ(N1, N2):
    return np.random.uniform(size=(N1,N2))

@jit
def network(x, JJ0, JJ1, gamma0, gamma1, gamma2, omega0, omega1, omega2):
    """
    The main function. Evaluate the scattering matrix, with input x.

    The network is of the type 784 (input pixels) -- N1 -- 10
    with fully connected layers.

    The matrix between layer 0(input) and layer 1 is JJ0 (shape (N0,N1)),
    and JJ1 is the matrix between layer 1 and layer 2.

    gamma0..2 are the vectors describing the decay rates in the various layers,
    and omega0..2 are the detunings.

    JJ0,JJ1,omega0..2 are all trainable.
    """
    # we consider a network where all layer 0 neurons are coherently illuminated
    # at the same strength and we collect the light at layer 2 (transmission setup)
    # this seems to be most natural if one wants to have a limit which matches
    # linear ANNs.
    chi2 = suscept(gamma2, -omega2)
    chi1_inv = invSuscept(gamma1, -omega1)
    JJ1_tr=jnp.transpose(JJ1)
    JJ0_tr=jnp.transpose(JJ0)
    chi1_tilde = jnp.linalg.inv( chi1_inv + jnp.matmul(jnp.matmul(JJ1, chi2), JJ1_tr) )
    chi0_inv = invSuscept(gamma0, -(x+omega0) )
    chi0_tilde = jnp.linalg.inv( chi0_inv + jnp.matmul(jnp.matmul(JJ0, chi1_tilde), JJ0_tr) )
    # the transmission scattering matrix, going from layer 0 to layer 2:
    return -jnp.dot(chi2, jnp.dot( JJ1_tr, jnp.dot(chi1_tilde, jnp.dot(JJ0_tr, jnp.dot(chi0_tilde, jnp.sqrt(gamma0))))))

@jit
def costF(x, y, JJ0, JJ1, gamma0, gamma1, gamma2, omega0, omega1, omega2):
    prob = jnp.log(jax.nn.softmax(8 * jnp.imag(network(x, JJ0, JJ1, gamma0, gamma1, gamma2, omega0, omega1, omega2))))
    return - jnp.dot(prob, y)

@jit
def costF_vmap(x, y, JJ0, JJ1, gamma0, gamma1, gamma2, omega0, omega1, omega2):
    return jnp.mean(vmap(costF, in_axes = (0,0,None,None,None,None,None,None,None,None), out_axes=0)(x, y, JJ0, JJ1, gamma0, gamma1, gamma2, omega0, omega1, omega2),axis=0)

@jit
def real_costF_grad_all_mean_and_value(x, y, JJ0, JJ1, gamma0, gamma1, gamma2, omega0, omega1, omega2):
    return value_and_grad(costF_vmap, argnums=[2,3,7,8,9])(x, y, JJ0, JJ1, gamma0, gamma1, gamma2, omega0, omega1, omega2)

"""# Loading and rescaling fashion MNIST"""

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)
x_train_vect = jnp.float32(jnp.array(input_scale * (1 - jnp.array(x_train)/256)))
x_test_vect = jnp.float32(jnp.array(input_scale * (1 - jnp.array(x_test)/256)))


if scale_down_images:
    # scaling down pictures
    n_train=np.shape(x_train)[0]
    x_train_new = np.empty((n_train,int(784/4)))
    n_test=np.shape(x_test)[0]
    x_test_new = np.empty((n_test,int(784/4)))

    row = np.arange(0,28,2)
    idx_00=[]
    for k in range(0,28,2):
        idx_00.extend(list(k*28+row))

    # indices for all 4 pixels to be grouped
    idx_00 = np.array(idx_00)
    idx_01 = idx_00 + 1
    idx_10 = idx_00 + 28
    idx_11 = idx_00 + 29

    # average over groups of 2x2
    x_train_new[:,:]=0.25*(1.0*x_train[:,idx_00]+1.0*x_train[:,idx_01]+
                    1.0*x_train[:,idx_10]+1.0*x_train[:,idx_11])

    x_test_new[:,:]=0.25*(1.0*x_test[:,idx_00]+1.0*x_test[:,idx_01]+
                    1.0*x_test[:,idx_10]+1.0*x_test[:,idx_11])

    # data vectors 'real'

    x_train_vect = jnp.float32(jnp.array(input_scale * (1 - jnp.array(x_train_new)/256)))

    x_test_vect = jnp.float32(jnp.array(input_scale * (1 - jnp.array(x_test_new)/256)))


# fill correct one-hot output
y_train = np.array(y_train, dtype = int)
y_train_vect = np.zeros((len(y_train),10))

for jj in range(len(y_train)):
    y_train_vect[jj, int(y_train[jj])]=1.0
y_train_vect = jnp.float32(y_train_vect)

y_test = np.array(y_test, dtype = int)
y_test_vect = np.zeros((len(y_test),10))
for jj in range(len(y_test)):
    y_test_vect[jj, int(y_test[jj])]=1.0
y_test_vect = jnp.float32(y_test_vect)

"""# Visualization and analysis routines"""

# visualize_test_batch
# to be used during training

# create a test batch that is nicely ordered
# first samples_per_fig will be '0' etc
samples_per_fig=5
pixels_per_fig=int(784/scale_image_factor)
x_test_batch=np.empty((samples_per_fig*10,pixels_per_fig),dtype='complex')

for j in range(10):
    indices=np.where(y_test==j)[0]
    x_test_batch[j*samples_per_fig:(j+1)*samples_per_fig,:]=x_test_vect[indices[:samples_per_fig],:]

x_test_batch=jnp.complex64(x_test_batch)

def visualize_test_batch_params(JJ0,JJ1,omega0,omega1,omega2):
    samples_per_fig=5
    out_network = jnp.ravel(jnp.imag(vmap(network, in_axes = (0,None,
            None,None,None,None,None,None,None), out_axes = 0)(x_test_batch, JJ0,
            JJ1, gamma0, gamma1, gamma2, omega0, omega1, omega2)))
    out_network = out_network.reshape(len(x_test_batch), 10)
    plt.figure(dpi=200)
    plt.imshow(np.transpose(out_network[:50]))
    for s in range(10):
        plt.plot((s*samples_per_fig-0.5,s*samples_per_fig-0.5),
                 (-0.5,9.5),color="white")
    plt.axis('off')

# get the whole confusion matrix
# on a larger part of the data set!

network_vmap=vmap(network, in_axes = (0,None,None,None,None,None,
                                      None,None,None), out_axes = 0)

conf_samples_per_fig=1000
pixels_per_fig=int(784/scale_image_factor)
x_test_batch_large=np.empty((conf_samples_per_fig*10,pixels_per_fig),dtype='complex')

for j in range(10):
    indices=np.where(y_test==j)[0]
    x_test_batch_large[j*conf_samples_per_fig:(j+1)*conf_samples_per_fig,:]=x_test_vect[indices[:conf_samples_per_fig],:]

x_test_batch_large=jnp.complex64(x_test_batch_large)

def get_confusion_matrix(params,samplesize=None):

    if samplesize is None:
        samplesize=conf_samples_per_fig
    JJ0=params[0]
    JJ1=params[1]
    omega0=params[2]
    omega1=params[3]
    omega2=params[4]

    acc_test_digit=np.zeros((10,10))
    for j in range(10):
        n_sub_samples=int(conf_samples_per_fig/samplesize)
        for sample_idx in range(n_sub_samples):
            the_set = x_test_batch_large[j*conf_samples_per_fig:j*conf_samples_per_fig+samplesize]
            out_network_test = jnp.ravel(jnp.imag(network_vmap(the_set, JJ0, JJ1, gamma0,
                                gamma1, gamma2, omega0, omega1, omega2)))
            out_network_test = out_network_test.reshape(len(the_set), 10)
            for k in range(10):
                acc_test_digit[j][k]+= sum(jnp.argmax(out_network_test, axis = 1) == k)/len(the_set)
        acc_test_digit[j]/=n_sub_samples
        print(acc_test_digit[j][j])

    return acc_test_digit

# plot the confusion matrix
def plot_confusion_matrix(name,confm):
    plt.matshow(confm)
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))
    plt.xlabel("detected")
    plt.ylabel("true")
    plt.title(name+" "+f"{np.mean(np.diag(confm)*100):2.1f}%")
    plt.colorbar()

def save_params(NAME, t, params):
    np.savez(NAME+"_t=" + str(t) + "_params.npz", *params)

def load_params(NAME, t):
    arrays=np.load(NAME+"_t=" + str(t) + "_params.npz")
    return [arrays[key] for key in arrays]

# get cost and grads, given both the trainable parameters
# and the fixed parameters
# --- this needs to be updated whenever you like a different
# mix of trainable and non-trainable

def restricted_cost_and_grad(x_batch,y_batch,params_fixed,params):
    # JJ0, JJ1, (gamma0, gamma1, gamma2,) omega0, omega1, omega2
    # the gammas in brackets are the fixed parameters in this case
    return real_costF_grad_all_mean_and_value(x_batch, y_batch, params[0],
                    params[1], *params_fixed, params[2], params[3], params[4])

def train(params_fixed,params,optimizer,opt_state,nsteps,cost_values):
    @jax.jit
    def gradient_step(x_batch,y_batch,params, opt_state):
        value, grads = restricted_cost_and_grad(x_batch,y_batch,params_fixed,params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    j=-1
    iterations_per_batch=60000/batch_size
    accuracies=[]

    for step in range(nsteps):
        if step%iterations_per_batch==0 or step==nsteps-1:
            save_params(DIRNAME+"/"+FILENAME,step,params)

        j+=1

        if j % iterations_per_batch == 0:
            random_mix_2 = np.random.choice(len(x_train_vect), len(x_train_vect), replace = False)
            j = 0

        batch_subset = random_mix_2[j * batch_size: (j+1) * batch_size]
        x_batch = x_train_vect[batch_subset]
        y_batch = y_train_vect[batch_subset]
        params,opt_state,value=gradient_step(x_batch,y_batch,params,opt_state)
        cost_values.append(float(value))
        if step%300==0 or step==nsteps-1:
            #clear_output(wait=True)
            #visualize_test_batch_params(*params)
            #plt.show()
            print(f"{step:5d}:  cost={value:2.3f}             ")

        if step%iterations_per_batch==0 or step==nsteps-1:
            conf_matrix=get_confusion_matrix(params, samplesize=100)

            plot_confusion_matrix(name,conf_matrix)
            plt.savefig(DIRNAME+"/"+FILENAME+"_confusion.pdf")
            plt.show()

            accuracy=np.mean(np.diag(conf_matrix))
            print("Epoch ", int(step/iterations_per_batch), " Accuracy: ", accuracy)
            accuracies.append(accuracy)

    print("Accuracies:")
    print(accuracies)

    return params,opt_state

# initialization with 'real' parameter types

N0 = int(784/scale_image_factor)
# N1 declared in beginning
N2 = 10

gamma_max = 1.0

omega0 = random_layer0_detuning_scale * (jnp.float32(randomVector(N0)) - 1)

gamma0 = gamma_max * jnp.float32(jnp.ones((1,N0))[0])
gamma1 = gamma_max * jnp.float32(jnp.ones((1,N1))[0])

omega1 = random_detuning_scale * (jnp.float32(randomVector(N1)) - .5)

gamma2 = gamma_max * jnp.float32(jnp.ones((1,N2))[0])

omega2 = random_detuning_scale * (jnp.float32(randomVector(N2)) - .5)

JJ0 = random_scale0 * jnp.sqrt(6/(N0+N1)) * jnp.float32(randomJJ(N0, N1) - .5)
JJ1 = random_scale1 *  jnp.sqrt(6/(N1+N2)) * jnp.float32(randomJJ(N1, N2) - .5)

DIRNAME="outputs"


name="results"
optimizer_func=optimizersched

print(name)

params_fixed=(gamma0, gamma1, gamma2)
params=(JJ0, JJ1, omega0, omega1, omega2)

optimizer=optimizer_func
opt_state=optimizer.init(params)

FILENAME=name

cost_values=[0.0]
params,opt_state=train(params_fixed,params,optimizer,opt_state,train_batches,cost_values)

# output cost values:

plt.plot(cost_values)
plt.title("cost "+name)
plt.savefig(DIRNAME+"/"+FILENAME+"_cost.pdf")
plt.show()

np.savetxt(DIRNAME+"/"+FILENAME+"_cost.csv", cost_values, delimiter=",")

# output confusion matrix:

conf_matrix=get_confusion_matrix(params, samplesize=100)

plot_confusion_matrix(name,conf_matrix)
plt.savefig(DIRNAME+"/"+FILENAME+"_confusion.pdf")
plt.show()

np.savetxt(DIRNAME+"/"+FILENAME+"_confusion.csv", conf_matrix, delimiter=",")

# output total accuracy

accuracy=np.array([np.mean(np.diag(conf_matrix))])
np.savetxt(DIRNAME+"/"+FILENAME+"_accuracy.csv", accuracy, delimiter=",")

end_time=time.time()

print("Run took ", end_time-start_time, " seconds.")
