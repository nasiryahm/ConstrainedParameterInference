is_cupy = True
try:
    import cupy as xp
except ImportError:
    print("Unable to load cupy. Falling back to numpy.")
    import numpy as xp
    is_cupy = False
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import getopt, os, sys, json
from copi.copi import *
from copi.utils import *
from keras.datasets import mnist, cifar10

device = 0
seed = 1
algo = 'bp'
nb_hid_layers = 4
nb_hid_units = 500
dataset = 'MNIST'
learning_rate = 1e-3
nb_epochs = 100
decorr = False
momentum = False
fa = False
# Setting up the options for simulation
opts, remaining = getopt.getopt(
    sys.argv[1:],
    '',
    ['seed=',
     'device=',
     'algo=',
     'nb_hid_layers=',
     'nb_hid_units=',
     'dataset=',
     'learning_rate=',
     'nb_epochs=',
     'decorr',
     'momentum',
     'fa'])
for opt, arg in opts:
    if opt == '--seed':
        seed = int(arg)
    if opt == '--device':
        device = int(arg)
    if opt == '--algo':
        algo = str(arg)
    if opt == '--nb_hid_layers':
        nb_hid_layers = int(arg)
    if opt == '--nb_hid_units':
        nb_hid_units = int(arg)
    if opt == '--dataset':
        dataset = str(arg)
    if opt == '--learning_rate':
        learning_rate = float(arg)
    if opt == '--nb_epochs':
        nb_epochs = int(arg)
    if opt == '--decorr':
        decorr = True
    if opt == '--momentum':
        momentum = True
    if opt == '--fa':
        fa = True
if is_cupy:
  xp.cuda.Device(device).use()

# Ensuring options
assert (algo == 'bp' or algo == 'copi'), "Choose either 'bp' or 'copi' as your algorithm."
assert (dataset == 'MNIST' or dataset == 'CIFAR10'), "Choose either 'MNIST' or 'CIFAR10' as your dataset."
if decorr:
  assert momentum == False, "Momentum cannot be combined with learned decorrelation."

net_struct = []
if dataset == 'MNIST':
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  net_struct.append(784)
else:
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  net_struct.append(3072)

# Reshaping to flat digits
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
y_test = xp.squeeze(y_test)
y_train = xp.squeeze(y_train)

# Creating onehot encoded targets
y_train_onehot = indices_to_onehot(y_train)
y_test_onehot = indices_to_onehot(y_test)

# Normalizing data
x_train = x_train / 255.0
x_test = x_test / 255.0

# If GPU, copy data to the device
if is_cupy:
  x_train = xp.asarray(x_train)
  y_train = xp.asarray(y_train)
  x_test = xp.asarray(x_test)
  y_test = xp.asarray(y_test)
  y_train_onehot = xp.asarray(y_train_onehot)
  y_test_onehot = xp.asarray(y_test_onehot)

for l in range(nb_hid_layers):
  net_struct.append(nb_hid_units)
net_struct.append(10)


net_measures = {'acc':{'train': [], 'test': []}, 'loss':{'train': [], 'test': []}}

batch_size = 50
nb_batches = int(len(x_train) / batch_size)
check_interval = 100
  
net_outpath = './results/' + dataset + '/' + algo 
if decorr:
  net_outpath += '_decorr'
if momentum:
  net_outpath += '_momentum'
if fa:
  net_outpath += '_fa'
net_outpath += '/' + str(seed) + '/'
os.makedirs(net_outpath, exist_ok=True)

print('\n Seed: ' + str(seed))
net = copi_net(net_struct, act_func=leaky_relu, der_act_func=der_leaky_relu, seed=seed, bp=(algo == 'bp'), decorr=decorr, momentum=momentum, fa=fa)
net.save_params(net_outpath + 'init_')

print("Epoch: ", end="", flush=True)
for epoch in range(nb_epochs):
  print(str(epoch) + "", end="", flush=True)
  for batch in range(nb_batches):
    input_data = x_train[batch*batch_size:(batch+1)*batch_size]
    target_data = y_train_onehot[batch*batch_size:(batch+1)*batch_size]

    # Carrying out a forward pass, error measure, and parameter update for CPI
    output_data = net.forward(input_data)
    net_output = net.act_func(output_data[-1])
    output_error = (target_data - net_output)*net.der_act_func(output_data[-1])
    custom_lr = learning_rate

    # After the first epoch, the learning rate for networks with decorrelation is increased
    if (epoch > 0) and decorr:
      custom_lr = 10.0*learning_rate
    net.update_parameters(output_data, output_error, custom_lr, ratio)

    # Every {check_interval} batches, we record the train/test acc/loss
    if (batch + epoch*nb_batches) % check_interval == 0:
      acc, loss = perf_check(net.act_func(net.forward(x_train)[-1]), y_train, y_train_onehot)
      net_measures['acc']['train'].append(acc); net_measures['loss']['train'].append(loss)
      acc, loss = perf_check(net.act_func(net.forward(x_test)[-1]), y_test, y_test_onehot)
      net_measures['acc']['test'].append(acc); net_measures['loss']['test'].append(loss)
  print(" (" + str(net_measures['acc']['train'][-1]) + "," + str(net_measures['acc']['test'][-1]) + "), ", end="", flush=True)

net.save_params(net_outpath)

set_to_plot = ['train', 'test']
measures = ['acc', 'loss']
for i, se in enumerate(set_to_plot):
  for j, me in enumerate(measures):  
    net_measures[me][se] = np.asarray(net_measures[me][se]).tolist()

net_file = open(net_outpath + '_measures.json', 'w')
json.dump(net_measures, net_file)
net_file.close()
