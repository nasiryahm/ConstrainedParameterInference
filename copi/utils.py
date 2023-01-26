import numpy as np
import torch
import torchvision
import os

def indices_to_onehot(data):
    '''
      Function for the conversion of labels to a onehot encoding
    '''
    nb_categories = np.max(data) + 1
    onehot = np.zeros((len(data), nb_categories))
    onehot[range(len(data)), data] = 1.0
    return onehot

def load_dataset(dataset_importer, device, fltype, validation):
  if dataset_importer == 'TIN':
    if os.path.exists('./datasets/tiny-imagenet-200/y_train.npy'):
      print('Loading TinyImageNet')
      x_train = np.load('./datasets/tiny-imagenet-200/x_train.npy')
      y_train = np.load('./datasets/tiny-imagenet-200/y_train.npy').astype(int)
      x_test = np.load('./datasets/tiny-imagenet-200/x_test.npy')
      y_test = np.load('./datasets/tiny-imagenet-200/y_test.npy').astype(int)
    else:
      print('Loading TinyImageNet')

      train_datasetpath = './datasets/tiny-imagenet-200/train/'
      test_datasetpath = './datasets/tiny-imagenet-200/val/images/'
      
      train_dataset = torchvision.datasets.ImageFolder(train_datasetpath)
      test_dataset = torchvision.datasets.ImageFolder(test_datasetpath)

      x_test = np.empty((len(test_dataset.targets), 3, 64, 64), dtype=np.float32)
      y_test = np.empty((len(test_dataset.targets)))
      for indx, (img, label) in enumerate(test_dataset.imgs):
        x_test[indx] = torchvision.transforms.ToTensor()(test_dataset.loader(img).convert("RGB"))
        y_test[indx] = label
      x_test = np.asarray(x_test); y_test = np.asarray(y_test)
      print('TinyImageNet test set loaded')

      np.save('./datasets/tiny-imagenet-200/x_test.npy', x_test)
      np.save('./datasets/tiny-imagenet-200/y_test.npy', y_test)

      x_train = np.empty((len(train_dataset.targets), 3, 64, 64), dtype=np.float32)
      y_train = np.empty((len(train_dataset.targets)))
      for indx, (img, label) in enumerate(train_dataset.imgs):
        x_train[indx] = torchvision.transforms.ToTensor()(train_dataset.loader(img).convert("RGB"))
        y_train[indx] = label
      x_train = np.asarray(x_train); y_train = np.asarray(y_train)
      print('TinyImageNet training set loaded')

      np.save('./datasets/tiny-imagenet-200/x_train.npy', x_train)
      np.save('./datasets/tiny-imagenet-200/y_train.npy', y_train)

  else:
    train_dataset = dataset_importer('./datasets/', train=True, download=True)
    test_dataset = dataset_importer('./datasets/', train=False, download=True)
  
    # Loading dataset
    x_train = train_dataset.data; y_train = train_dataset.targets
    x_test = test_dataset.data; y_test = test_dataset.targets

  # Reshaping to flat digits
  x_train = x_train.reshape(x_train.shape[0], -1)
  x_test = x_test.reshape(x_test.shape[0], -1)
  y_train = np.asarray(y_train); y_test = np.asarray(y_test)

  # Extracting a validation, rather than test, set
  # Last 10K samples taken as test
  if validation and not (dataset_importer == 'tinyimagenet'):
    x_test = x_train[-10000:]
    y_test = y_train[-10000:]

    x_train = x_train[:50000]
    y_train = y_train[:50000]

  # Squeezing out any excess dimension in the labels (true for CIFAR10/100)
  y_train = np.squeeze(y_train)
  y_test = np.squeeze(y_test)

  y_train = torch.tensor(y_train); y_test = torch.tensor(y_test); 
  x_train = torch.tensor(x_train); x_test = torch.tensor(x_test); 
 
  # Creating onehot encoded targets
  y_train_onehot = torch.nn.functional.one_hot(y_train, torch.max(y_train)+1)
  y_test_onehot = torch.nn.functional.one_hot(y_test, torch.max(y_train)+1)

  # Normalizing data
  x_train = x_train / 255.0
  x_test = x_test / 255.0

  # Data to device (datasets small enough to fit directly)
  x_train = x_train.to(device).type(fltype)
  y_train = y_train.to(device).type(fltype)
  y_train_onehot = y_train_onehot.to(device).type(fltype)

  x_test = x_test.to(device).type(fltype)
  y_test = y_test.to(device).type(fltype)
  y_test_onehot = y_test_onehot.to(device).type(fltype)

  return x_train, y_train, y_train_onehot, x_test, y_test, y_test_onehot

def perf_check(outputs, labels, onehots):
  '''
    Returns the accuracy and MSE loss between some network output and targets
    
    Parameters:
      outputs (xp array, floats): Network outputs of shape Nx10
      labels (xp array, ints): Sample labels as integers of shape N
      onehots (xp array, floats): Onehot representation of labels to labels of shape Nx10

    Returns:
      acc (float): Accuracy computed as fraction where max output equivalent to label
      loss (float): Average loss computed as MSE between outputs and onehots
  '''
  acc = ((torch.argmax(outputs.data, 1) == labels).sum() / len(outputs)).item()
  loss = ((outputs - onehots)**2).mean().item()
  return acc, loss

def measure_perf(this_model, datasets, train_test):
  perf_accum = 0
  loss_accum = 0
  input_data = datasets['x_' + train_test]
  labels = datasets['y_' + train_test]
  target_data = datasets['y_'+ train_test + '_onehot']
  
  batch_size = this_model.sim_params['batch_size']
  n_batches = int(len(input_data) / batch_size)
  for b in range(n_batches):
    b_input_data = input_data[b*batch_size:(b+1)*batch_size]
    b_target_data = target_data[b*batch_size:(b+1)*batch_size]
    b_labels = labels[b*batch_size:(b+1)*batch_size]

    x_vals, a_vals, net_output = this_model(b_input_data)
    perf, loss = perf_check(net_output, b_labels, b_target_data)

    perf_accum += perf
    loss_accum += loss

  this_model.sim_params['performances'][train_test]['loss'].append(loss_accum/n_batches)
  this_model.sim_params['performances'][train_test]['acc'].append(perf_accum/n_batches);


def training_loop(this_model, datasets, n_epochs=1, verbose=True, high_density_measures=False):
  batch_size = this_model.sim_params['batch_size']
  nb_batches = int(len(datasets['x_train']) / batch_size)

  eval_interval = nb_batches
  if high_density_measures:
    eval_interval = int(nb_batches / 10)

  if verbose: print("Epoch: (train acc, train loss)")
  for e in range(n_epochs):
    if verbose: print(str(e) + " ", end="")

    if e == 0:
      this_model.sim_params['update_W'] = False
    else:
      this_model.sim_params['update_W'] = True

    for b in range(nb_batches):
      input_data = datasets['x_train'][b*batch_size:(b+1)*batch_size]
      target_data = datasets['y_train_onehot'][b*batch_size:(b+1)*batch_size]

      x_vals, a_vals, net_output = this_model(input_data)
      perts = this_model.perturbations(x_vals, a_vals, net_output, target_data)
      this_model.update_params(x_vals, a_vals, perts)

      if ((b + 1) % eval_interval) == 0:
        measure_perf(this_model, datasets, 'train')
        measure_perf(this_model, datasets, 'test')
        if verbose: print(str(this_model.sim_params['performances']['train']['acc'][-1]) + ", " + 
                        str(this_model.sim_params['performances']['train']['loss'][-1]) + " | " +
                        str(this_model.sim_params['performances']['test']['acc'][-1]) + ", " + 
                        str(this_model.sim_params['performances']['test']['loss'][-1]) + " |")

    # Shuffle Data
    permutation = np.random.permutation(len(datasets['x_train']))
    datasets['x_train'] = datasets['x_train'][permutation]
    datasets['y_train'] = datasets['y_train'][permutation]
    datasets['y_train_onehot'] = datasets['y_train_onehot'][permutation]
