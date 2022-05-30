is_cupy = True
try:
    import cupy as xp
except ImportError:
    print("Unable to load cupy. Falling back to numpy.")
    import numpy as xp
    is_cupy = False
import numpy as np

class copi_layer():
  '''
  Class encompassing a single feedforward layer of a DNN with COPI learning rule
  '''
  def __init__(self, nb_inputs, nb_outputs, act_func, der_act_func, bp, decorr, fa, momentum):
    # Storing layer shape
    self.shape = (nb_inputs, nb_outputs)
    # Forward weight matrix (Xavier/Glorot normal init)
    self.W = xp.sqrt(2/(nb_inputs+nb_outputs))*xp.random.randn(nb_outputs, nb_inputs)
    if decorr:
      # Lateral (inhibitory/decorrelating) weight matrix
      self.R = xp.eye(nb_inputs)
    if fa:
      # Random feedback matrix for feedback alignment
      self.B = xp.sqrt(2/(nb_inputs+nb_outputs))*xp.random.randn(nb_outputs, nb_inputs)
    # Identity matrix of shape inputs, useful for whitening
    self.eye = xp.eye(nb_inputs)
    # Activation function (applied to inputs for convenience)
    self.act_func = act_func
    # Derivative of activation function
    self.der_act_func = der_act_func
    
    self.bp = bp
    self.decorr = decorr
    self.fa = fa

    self.momentum = momentum
    if self.momentum and self.decorr:
      # if momentum with decorrelation, instead do weight scaling by variance
      self.mom = xp.ones([nb_inputs])
    else:
      self.first_moment_W = xp.zeros(self.W.shape)
      self.second_moment_W = xp.zeros(self.W.shape)
  
  def __call__(self, data):
    return self.forward(data)

  def proc_inputs(self, input_data):
    '''
    Returns only the activation function passed, whitened input

    Parameters:
      input_data (xp array, floats): Layer input data of shape NxM (batches X input shape)
    Returns:
      data (xp array, floats): G (I-M) f(input_data)
    '''
    assert input_data.shape[1] == self.shape[0], 'Dimension of inputs incorrect'
    if self.decorr:
      return xp.einsum('ij, nj->ni', self.R, self.act_func(input_data))
    else:
      return self.act_func(input_data)

  
  def forward(self, input_data):
    '''
    Returns the layer transformation of a batch of data

    Parameters:
      input_data (xp array, floats): Layer input data of shape NxM (batches X input shape)
    Returns:
      data (xp array, floats): W G (I-M) f(input_data)
    '''
    assert input_data.shape[1] == self.shape[0], 'Dimension of inputs incorrect'
    # Lateral whitening process
    h = self.proc_inputs(input_data)
    # Forward transformation through the layer
    data = xp.einsum('ij,nj->ni', self.W, h)
    assert not xp.any(xp.isnan(data)), 'NaN data'
    return data

  
  def bp_error(self, input_data, error):
    '''
    Backpropagates an error vector through a layer, given some input data

    Parameters:
      input_data (xp array, floats): Layer input data of shape NxM (batches X input shape)
      error (xp array, floats): Error values for each output unit of shape NxP (batches x output shape)
    Returns:
      error (xp array, floats): error W G (I - M)
    '''
    assert input_data.shape[1] == self.shape[0], 'Dimension of inputs incorrect'
    assert error.shape[1] == self.shape[1], 'Dimension of errors incorrect'
    assert input_data.shape[0] == error.shape[0], 'Input/Error batch dimension mismatch'

    if self.fa:
      error = xp.einsum('nj,ji->ni', error, self.B)
    else:
      error = xp.einsum('nj,ji->ni', error, self.W)
      if self.decorr:
        error = xp.einsum('nj,ji->ni', error, self.R)
    error = error*self.der_act_func(input_data)
    return error
  
  def update_parameters_decorr(self, h, learning_rate, ratio):
    corr = (1/len(h))*xp.einsum('ni,nj->ij', h, h)*(1.0 - self.eye)
    # Updating the lateral decorrelating weights by iterative whitening
    r_update = -ratio*((xp.einsum('ij,jk->ik', corr, self.R)))
    if self.momentum:
      r_update *= self.mom
      self.mom += 0.01*xp.mean(1.0 - self.mom*(1e-5 + h**2), axis=0)
    self.R += learning_rate*r_update
  
  def update_parameters(self, input_data, pert_output_data, learning_rate, ratio=1.0):
    '''
    Update parameters W and (perhaps) R -- no return.

    Parameters:
      input_data (xp array, floats): Layer input data of shape NxM (batches X input shape)
      pert_output_data (xp array, floats): Perturbed output data NxP (batches X output shape)
      learning_rate (float): Learning rate value
      ratio (float): A multiplication ratio which modulates the whitening learning rule
    '''
    assert input_data.shape[1] == self.shape[0], 'Dimension of inputs incorrect'
    assert pert_output_data.shape[1] == self.shape[1], 'Dimension of errors incorrect'
    assert input_data.shape[0] == pert_output_data.shape[0], 'Input/Error batch dimension mismatch'
    
    h = self.proc_inputs(input_data)
    if self.decorr: 
      self.update_parameters_decorr(h, learning_rate, ratio)
    
    if self.bp:
      # Updating weights by BP
      w_update = (1/len(input_data))*(xp.einsum('ni,nj->ij', pert_output_data, h) - xp.einsum('ni,nj->ij', h @ self.W.transpose(), h))
      self.update_W(w_update, learning_rate)
    else:
      # Updating weights by COPI
      w_update = (1/len(input_data))*(xp.einsum('ni,nj->ij', pert_output_data, h) - xp.einsum('nj,ij->ij', h**2, self.W))
      self.update_W(w_update, learning_rate)

  def update_W(self, w_update, learning_rate):
    if self.momentum and not self.decorr:
      self.first_moment_W = 0.1*w_update + 0.9*self.first_moment_W
      self.second_moment_W = 0.01*(w_update**2) + 0.99*self.second_moment_W
      learning_rate = (learning_rate / (1e-8 + xp.sqrt(self.second_moment_W)))
      self.W += learning_rate*self.first_moment_W
    elif self.momentum and self.decorr:
      self.W += learning_rate*self.mom*w_update
    else:
      self.W += learning_rate*w_update

  def save_params(self, path):
    xp.save(path + "W.npy", self.W)
    if self.decorr:
      xp.save(path + "R.npy", self.R)
    if self.fa:
      xp.save(path + "B.npy", self.B)

  def load_params(self, path):
    self.W = xp.load(path + "W.npy")
    if self.decorr:
      self.R = xp.load(path + "R.npy")
    if self.fa:
      self.B = xp.load(path + "B.npy")

class copi_net():
    '''
    Class constructing multi-layer networks from copi_layer instances
    '''
    def __init__(self, net_structure, act_func=lambda x: x, der_act_func=lambda x: xp.ones(x.shape), seed=42, bp=True, decorr=False, fa=False, momentum=False):
      xp.random.seed(seed)
      self.net_structure = net_structure
      self.layers = []
      self.act_func = act_func
      self.der_act_func = der_act_func

      for i in range(len(net_structure) - 1):
        self.layers.append(copi_layer(net_structure[i], net_structure[i+1], act_func, der_act_func, bp, decorr, fa, momentum))

    def forward(self, input_data):
      data = [xp.copy(input_data)]
      for layer in self.layers:
        state = layer.forward(data[-1])
        data.append(xp.copy(state))
        assert not xp.any(xp.isnan(state)), 'NaN data'
      return data
    
    def update_parameters(self, output_data, output_error, learning_rate, ratio=1.0):
      error = output_error
      for i, layer in enumerate(self.layers[::-1]):
        layer.update_parameters(output_data[-i-2], output_data[-i-1] + error, learning_rate, ratio=ratio)
        error = layer.bp_error(output_data[-i-2], error)
    
    def save_params(self, path):
      for i, layer in enumerate(self.layers):
        layer.save_params(path + str(i))
    
    def load_params(self, path):
      for i, layer in enumerate(self.layers):
        layer.load_params(path + str(i))
