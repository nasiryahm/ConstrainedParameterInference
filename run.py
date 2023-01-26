import numpy as np
import copy
import yaml
import copi
import torchvision
import torch
import getopt, sys

seed = 1
dataset = 'MNIST'
config = 'copi_bp_boosted'
validation = False
high_density_measures = False
# Setting up the options for simulation
opts, remaining = getopt.getopt(
    sys.argv[1:],
    '',
    ['seed=',
     'dataset=',
     'config=',
     'validation',
     'high_density_measures'
     ])
for opt, arg in opts:
    if opt == '--seed':
        seed = int(arg)
    if opt == '--dataset':
        dataset = str(arg)
    if opt == '--config':
        config = str(arg)
    if opt == '--validation':
        validation = True
    if opt == '--high_density_measures':
        high_density_measures = True
assert dataset in ['MNIST', 'PerformanceTesting', 'CIFAR10', 'TIN'], 'Unsupported dataset!'

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# Setting floating point precision and device
fltype = torch.float32
torch.set_default_dtype(fltype)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Turning off all gradient computation for pytorch (not used in models)
torch.set_grad_enabled(False)

# Load desired dataset
dataset_func = None
if dataset == 'MNIST': dataset_func = torchvision.datasets.MNIST
elif dataset == 'CIFAR10' or dataset =='PerformanceTesting': dataset_func = torchvision.datasets.CIFAR10
else: dataset_func = 'TIN'

x_train, y_train, y_train_onehot, x_test, y_test, y_test_onehot = copi.load_dataset(dataset_func, device, fltype, validation=validation)

datasets = {
    'x_train': x_train, 'y_train': y_train, 'y_train_onehot': y_train_onehot,
    'x_test': x_test, 'y_test': y_test, 'y_test_onehot': y_test_onehot}

# Load config, and seed
with open("configs/" + dataset + "/" + config + ".yml", 'r') as stream:
    sim_params = yaml.safe_load(stream)
sim_params['seed'] = seed

print(dataset, sim_params['outpath'])

# Run networks
net = copi.NN_Builder(sim_params)
_ = net.to(device)

# Getting initial perf
copi.measure_perf(net, datasets, 'train')
copi.measure_perf(net, datasets, 'test')

copi.training_loop(net, datasets, n_epochs=100, high_density_measures=high_density_measures)

net.save()



