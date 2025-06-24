from gymnasium import Env, Wrapper, error, logger
from omni.isaac.lab.envs import ManagerBasedRLEnv
import torch

class Homeokinesis:
  """ Homekinetic controller """

  def __init__(self, agent_cfg):
        # print(agent_cfg)
        # exit()
        self.conf = agent_cfg['config']
        self.num_inputs = agent_cfg['num_inputs']
        self.num_outputs = agent_cfg['num_outputs']
        self.num_envs = agent_cfg['num_envs']
        
        # TODO allow other cuda numbers
        if torch.cuda.is_available() and agent_cfg['config']['device'] == 'cuda:0': 
          dev = "cuda:0" 
        else: 
          dev = "cpu"
        self.device = torch.device(dev)
        with self.device:
          self.A = self.eyes(self.num_envs, self.num_inputs, self.num_outputs)
          self.S = self.eyes(self.num_envs, self.num_inputs, self.num_inputs)
          self.C = self.eyes(self.num_envs, self.num_outputs, self.num_inputs)
          self.b = torch.zeros(self.num_envs, self.num_inputs, 1)
          self.h = torch.zeros(self.num_envs, self.num_outputs, 1)
          self.L = torch.zeros(self.num_envs, self.num_inputs, self.num_outputs)
          self.v_avg = torch.zeros(self.num_envs, self.num_inputs, 1)
          self.A_native = self.eyes(self.num_envs, self.num_inputs, self.num_outputs)
          self.C_native = self.eyes(self.num_envs, self.num_outputs, self.num_inputs)
          self.R = torch.zeros(self.num_envs, self.num_inputs, self.num_inputs)
          self.C *= self.conf['init_feedback_strength']
          self.S *= 0.05
          self.C_native *= 1.2
          self.y_teaching = torch.zeros(self.num_envs, self.num_outputs,1)
          self.x = torch.zeros(self.num_envs, self.num_inputs,1)
          self.x_smooth = torch.zeros(self.num_envs, self.num_inputs,1)
          self.x_buffer = torch.zeros(self.num_envs, self.conf['buffersize'], self.num_inputs, 1)
          self.y_buffer = torch.zeros(self.num_envs, self.conf['buffersize'], self.num_outputs, 1)
        self.t = 0


  def step(self, x):
      # adding noise to sensors
      noise = torch.normal(0, 0.05, size=(self.num_envs, self.num_inputs), device=self.device)
      x += noise

      # learn
      self.learn()
      # filling buffers
      steps_for_averaging: int = max(1, min(self.conf['steps_for_averaging'], self.conf['buffersize']))
      x = x.reshape(self.num_envs, self.num_inputs, 1)
      if steps_for_averaging > 1:
         self.x_smooth += (x - self.x_smooth) * (1.0 / steps_for_averaging)
      else:
         self.x_smooth = x
      self.x_buffer[:, self.t % self.conf['buffersize']] = self.x_smooth
      # forward pass
      y = torch.tanh(torch.matmul(self.C, self.x_smooth + self.v_avg * self.conf['creativity']) + self.h)
      # adding noise to actions
      noise = torch.normal(0, 0.05, size=(self.num_envs, self.num_outputs, 1), device=self.device)
      y += noise

      self.y_buffer[:, self.t % self.conf['buffersize']] = y
      self.t += 1
      # print(y[0])
      return torch.squeeze(y)
  
  def learn(self):
    steps_for_delay: int = max(1, min(self.conf['steps_for_delay'], self.conf['buffersize']-1))
    x       = self.x_buffer[:, (self.t - max(steps_for_delay, 1) + self.conf['buffersize']) % self.conf['buffersize']]
    y_creat = self.y_buffer[:, (self.t - max(steps_for_delay, 1) + self.conf['buffersize']) % self.conf['buffersize']]
    x_fut   = self.x_buffer[:, self.t % self.conf['buffersize']] # future sensor (with respect to x, y)
    xi      = x_fut - (torch.matmul(self.A, y_creat) + self.b + torch.matmul(self.S, x)) # here we use creativity
    z       = torch.matmul(self.C, x) + self.h # here no creativity
    y       = torch.tanh(z)

    g_prime = 1 - torch.tanh(z).pow_(2) # tanh derivative
    # TODO can this be done pythorchically
    g_prime_diag = torch.zeros(self.num_envs, self.num_outputs, self.num_outputs, device=self.device)
    for i, g in enumerate(g_prime):
       g_prime_diag[i] = torch.diagflat(g)

    L = torch.matmul(self.A, torch.matmul(g_prime_diag, self.C)) + self.S
    eta = torch.matmul(torch.linalg.pinv(self.A), xi)
    y_hat = y + eta * float(self.conf['cause_aware'])

    Lplus = torch.linalg.pinv(L)
    v     = torch.matmul(Lplus, xi)
    Lplus_t = torch.einsum('ijk->ikj', Lplus) # transposing each matrix
    chi   = torch.matmul(Lplus_t, v)

    mu = torch.einsum('ijk->ikj', self.A) # transposing each matrix
    mu = torch.matmul(g_prime_diag, mu)
    mu = torch.matmul(mu, chi)

    epsrel_temp = torch.matmul(self.C, v)
    # TODO can this be done pythorchically
    epsrel = torch.zeros(self.num_envs, self.num_outputs, self.num_outputs, device=self.device)
    for i, g in enumerate(epsrel_temp):
       epsrel[i] = torch.diagflat(g)
    epsrel = torch.matmul(epsrel, mu) * self.conf['sense'] * 2.0

    v_hat = v + x * self.conf['harmony']
    self.v_avg += (v - self.v_avg) * 0.1

    EE = 1.0
    #TODO add loga
    if self.conf['loga']:
      EE = 0.1 / (torch.linalg.norm(v, ord=2, dim=1) + 0.001)
      EE = EE.view(-1, 1, 1)
    x_t = torch.einsum('ijk->ikj', x)

    if self.conf['epsilon_A'] > 0:
      epsilon_S = self.conf['epsilon_A'] * self.conf['factor_S']
      epsilon_b = self.conf['epsilon_A'] * self.conf['factor_b']
      y_hat_t = torch.einsum('ijk->ikj', y_hat)
      self.A += torch.clamp(torch.matmul(xi, y_hat_t) * self.conf['epsilon_A'], -0.01, 0.01)
      if self.conf['damping'] > 0:
        self.A += torch.clamp(torch.pow(self.A_native - self.A, 3) * self.conf['damping'], -0.1, 0.1)
      if self.conf['use_extended_model']:
        self.S += torch.clamp(torch.matmul(xi, x_t) * epsilon_S + (self.S * (-self.conf['damping'] * 10)), -0.1, 0.1)
      self.b += torch.clamp(xi * epsilon_b + (self.b * - self.conf['damping']), -0.1, 0.1)


    if self.conf['epsilon_C'] > 0:
      v_hat_t = torch.einsum('ijk->ikj', v_hat)
      # TODO can this be done pythorchically
      y_diag = torch.zeros(self.num_envs, self.num_outputs, self.num_outputs, device=self.device)
      for i, g in enumerate(y):
        y_diag[i] = torch.diagflat(g)

      self.C += torch.clamp((torch.matmul(mu, v_hat_t) 
                              - torch.matmul(
                                 torch.matmul(y_diag, epsrel),
                                 x_t))
                             * (EE * self.conf['epsilon_C']),
                            -0.05, 0.05)
      if self.conf['damping'] > 0:
         self.C += torch.clamp(torch.pow(self.C_native - self.C, 3) * self.conf['damping'],
                               -0.05, 0.05)
      self.h += torch.clamp((mu * self.conf['harmony'] - torch.matmul(y_diag, epsrel)) 
                                * (EE * self.conf['epsilon_C'] * self.conf['factor_h']),
                               -0.05, 0.05)




  def eyes(self, n_instances, n_rows, n_cols):
    x = torch.eye(n_rows, n_cols)
    x = x.reshape((1, n_rows, n_cols))
    return x.repeat(n_instances, 1, 1)