import torch
import yaml
import numpy as np
import os

ALPHA=0.1
class NN_Builder(torch.nn.Module):
  def __init__(self, sim_params, act_func=lambda x: torch.where(x >= 0, x, ALPHA*x), der_act_func=lambda x: torch.where(x >= 0, 1.0, ALPHA)):
    super().__init__()
    assert sim_params['credit_algo'] in ['bp', 'fa'], "You have chosen an unsupported credit assignment algorithm"
    assert sim_params['learning_rule'] in ['bp', 'copi'], "You have chosen an unsupported learning rule"
    self.net_structure = sim_params['net_structure']
    self.decor_layers = []
    self.fwd_layers = []
    if sim_params['credit_algo'] == 'fa': self.bwd_layers = []
    if sim_params['adam'] == True: 
      self.adam_m = []
      self.adam_v = []
    self.eyes = []
    torch.manual_seed(sim_params['seed'])

    for indx in range(len(self.net_structure) - 1):
      self.decor_layers.append(torch.nn.Linear(self.net_structure[indx], self.net_structure[indx], bias=False))
      torch.nn.init.eye_(self.decor_layers[-1].weight)
      self.eyes.append(torch.nn.parameter.Parameter(torch.eye(self.net_structure[indx])))

      self.fwd_layers.append(torch.nn.Linear(self.net_structure[indx], self.net_structure[indx + 1], bias=False))
      torch.nn.init.xavier_normal_(self.fwd_layers[-1].weight)
      
      if sim_params['credit_algo'] == 'fa':
        self.bwd_layers.append(torch.nn.Linear(self.net_structure[indx+1], self.net_structure[indx], bias=False))
        torch.nn.init.xavier_normal_(self.bwd_layers[-1].weight)
      
      if sim_params['adam'] == True: 
        self.adam_m.append(torch.nn.parameter.Parameter(torch.zeros(self.net_structure[indx + 1], self.net_structure[indx])))
        self.adam_v.append(torch.nn.parameter.Parameter(torch.zeros(self.net_structure[indx + 1], self.net_structure[indx])))


    # This allows pytorch to manage moving everything to device
    self.decors_mod = torch.nn.ModuleList(self.decor_layers)
    self.fwd_mod = torch.nn.ModuleList(self.fwd_layers)
    self.eye_params = torch.nn.ParameterList(self.eyes)
    if sim_params['adam']:
       self.adam_m_params = torch.nn.ParameterList(self.adam_m)
       self.adam_v_params = torch.nn.ParameterList(self.adam_v)
    if sim_params['credit_algo'] == 'fa': self.bwd_mod = torch.nn.ModuleList(self.bwd_layers)

    # NOTE: Definition of derivative of leaky relu was incorrect
    self.act_func = act_func
    self.der_act_func = der_act_func
    self.sim_params = sim_params

  def forward(self, input_data):
    y = torch.clone(input_data)

    x_vals = []
    a_vals = []
    for indx in range(len(self.net_structure) - 1):
      x = self.decor_layers[indx](y)
      x_vals.append(torch.clone(x))

      a = self.fwd_layers[indx](x)
      a_vals.append(torch.clone(a))

      # Add activation function after every layer
      y = self.act_func(a)

    return x_vals, a_vals, a
  
  def update_params(self, x_vals, a_vals, perturbations):
    for indx in range(len(self.net_structure) - 1):
      # Update parameters
      if self.sim_params['update_W'] == True:
        # Forward Update Rule
        if self.sim_params['learning_rule'] == 'bp':
          dW = (1/len(x_vals[0]))*torch.einsum('ni,nj->ij', perturbations[indx], x_vals[indx])
        elif self.sim_params['learning_rule'] == 'copi':
          dW = (1/len(x_vals[0]))*(torch.einsum('ni,nj->ij', a_vals[indx] + perturbations[indx], x_vals[indx])) - torch.einsum('j,ij->ij', torch.mean(x_vals[indx]**2, axis=0), self.fwd_layers[indx].weight)
        if 'var_scaled' in self.sim_params.keys():
          if self.sim_params['var_scaled']:
            #dW = (1/len(x_vals[0]))*(torch.einsum('ni,nj->ij', (a_vals[indx] + perturbations[indx]), (1/(x_vals[indx]**2 + 1e-5))*x_vals[indx]) - (1/(1+1e-5))*self.fwd_layers[indx].weight)
            dW = (1/(torch.mean(x_vals[indx]**2, axis=0) + 1e-5))*dW
        if self.sim_params['adam']:
          dW = self.update_adam_variables(dW, indx)
        
        if 'regularizer' in self.sim_params.keys():
          dW -= self.sim_params['regularizer']*self.fwd_layers[indx].weight
        self.fwd_layers[indx].weight = torch.nn.parameter.Parameter(self.fwd_layers[indx].weight + 
                                                                    (self.sim_params['lr']) * dW)
        
      if self.sim_params['update_R'] == True:
        # Decorrelation Update Rule
        corr = (1/len(x_vals[0]))*torch.einsum('ni,nj->ij', x_vals[indx], x_vals[indx])*(1.0 - self.eyes[indx])
        if 'var_scaled' in self.sim_params.keys():
          if self.sim_params['var_scaled']:
            corr = (1/(torch.mean(x_vals[indx]**2, axis=0) + 1e-5))*corr
        if self.sim_params['bio_decorr'] == True:
          dR = -(torch.einsum('ij,jk->ik', self.decor_layers[indx].weight, corr))
        else:
          dR = -(torch.einsum('ij,jk->ik', corr, self.decor_layers[indx].weight))
        self.decor_layers[indx].weight = torch.nn.parameter.Parameter(self.decor_layers[indx].weight +
                                                                      (self.sim_params['lr']) * dR)

  def update_adam_variables(self, dW, indx):
    beta_1, beta_2, epsilon = self.sim_params['beta_1'], self.sim_params['beta_2'], self.sim_params['epsilon']
    self.adam_m[indx] = torch.nn.parameter.Parameter(beta_1*self.adam_m[indx] + (1.0 - beta_1)*dW)
    self.adam_v[indx] = torch.nn.parameter.Parameter(beta_2*self.adam_v[indx] + (1.0 - beta_2)*(dW**2))

    m_hat = (self.adam_m[indx] / (1.0 - beta_1))
    v_hat = (self.adam_v[indx] / (1.0 - beta_2))
    update =  m_hat / (torch.sqrt(v_hat) + epsilon)
    return update

  def perturbations(self, *args):
    if self.sim_params['credit_algo'] == 'bp':
      return self.bp_perturbations(*args)
    if self.sim_params['credit_algo'] == 'fa':
      return self.fa_perturbations(*args)

  def bp_perturbations(self, x_vals, a_vals, net_output, targets):
    perturbations = []

    pert = self.sim_params['loss_multiplier']*(targets - net_output)
    pert *= self.der_act_func(net_output)
    perturbations.append(torch.clone(pert))

    for indx in range(len(self.net_structure) - 2):
      pert = pert @ self.fwd_layers[-1-indx].weight
      pert = pert @ self.decor_layers[-1-indx].weight
      pert *= self.der_act_func(a_vals[-2-indx])
      perturbations.append(torch.clone(pert))
    
    return perturbations[::-1]

  def fa_perturbations(self, x_vals, a_vals, net_output, targets):
    perturbations = []

    pert = self.sim_params['loss_multiplier']*(targets - net_output)
    pert *= self.der_act_func(net_output)
    perturbations.append(torch.clone(pert))

    for indx in range(len(self.net_structure) - 2):
      pert = self.bwd_layers[-1-indx](pert)
      pert *= self.der_act_func(a_vals[-2-indx])
      perturbations.append(torch.clone(pert))
    
    return perturbations[::-1]
  
  def save(self):
    path = self.sim_params['outpath'] + "/" + str(self.sim_params['seed']) + "/"
    os.makedirs(path, exist_ok=True)
    # Saving configuration
    with open(path + 'sim_params.yml', 'w') as yaml_file:
      yaml.dump(self.sim_params, yaml_file, default_flow_style=False)
    # Saving model params
    torch.save(self.state_dict(), path + 'model_params.pth')
