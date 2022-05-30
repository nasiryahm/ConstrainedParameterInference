is_cupy = True
try:
    import cupy as xp
except ImportError:
    print("Unable to load cupy. Falling back to numpy.")
    import numpy as xp
    is_cupy = False
import numpy as np

def leaky_relu(x, alpha=0.1):
  '''
    Leaky ReLU function, returns scaled input data

    Parameters:
      x (xp array): Input data array (assumed NxM)
      alpha (float): Leak term (default 0.1)

    Returns:
      y (xp array): A copy of x with all negative values multiplied by alpha
  '''
  y = xp.copy(x)
  y[x < 0.0] = alpha*y[x < 0.0]
  return y


def der_leaky_relu(x, alpha=0.1):
  '''
    Derivative of Leaky ReLU function, returns array

    Parameters:
      x (xp array): Input data array (assumed NxM)
      alpha (float): Leak term (default 0.1)

    Returns:
      y (xp array): Array equivalent to x in shape with 1.0/alpha in place of +ve/-ve values
  '''
  y = xp.copy(x)
  y[x >= 0.0] = 1.0
  y[x < 0.0] = alpha
  return y

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
  acc = xp.mean(xp.argmax(outputs, axis=1) == labels)
  loss = xp.mean((outputs - onehots)**2)
  if is_cupy:
    return acc.get(), loss.get()
  return acc, loss

def indices_to_onehot(labels, nb_categories=10):
    '''
      Function for the conversion of 1D array of labels to a 2D onehot encoding
    '''
    onehot = np.zeros((len(labels), nb_categories))
    onehot[range(len(labels)), labels] = 1.0
    return onehot
