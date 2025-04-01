#!/usr/bin/env python
# coding: utf-8

# In[375]:


#reset -f


# In[376]:


import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt 

from pyDOE import lhs 


# In[377]:


torch.manual_seed(1234)
np.random.seed(1234)


# In[378]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Working on {device}")


# In[379]:


def batch_generator(x, t, y, batch_size, dev = device):   
    idx = torch.arange(len(x)) 
    
    # Ensure the batch size is not larger than the available data points
    num_batches = (len(x) + batch_size - 1) // batch_size  # calculate number of batches
    
    for i in range(num_batches): 
        
        batch_idx = idx[i * batch_size : min((i + 1) * batch_size, len(x))]  # Get batch indices
        
        batch_x = x[batch_idx].to(dev)
        batch_t = t[batch_idx].to(dev)
        batch_y = y[batch_idx].to(dev)
        
        # Yield the batch
        yield batch_x, batch_t, batch_y


# In[380]:


#functions for data handling
def normalize(inputs): # to [-1, 1]
    inputs_min, inputs_max = inputs.min(), inputs.max()
    
    return 2*(inputs - inputs_min) / (inputs_max - inputs_min) - 1


def denormalize(inputs): # from [-1, 1]
    inputs_min, inputs_max = inputs.min(), inputs.max()
    
    return ((inputs + 1)/2)*(inputs_max - inputs_min) + inputs_min


def standardize(inputs):
    
    return  (inputs - inputs.mean())/(inputs.std())


def destandardize(inputs):
    
    return  inputs*inputs.std() + inputs.mean()


# In[381]:


#Gets the untouched datas and conditions points numbers and returns the handled data ready for training
def data_handler(x, t, y, n_ic, n_bc, n_domain):

    #Prepare shuffled data and respective domain (initial and boundary conditions are shuffled when they're created)
    idx = torch.randperm(len(x))

    x_shuff = x[idx]
    t_shuff = t[idx]
    X_shuff, T_shuff = np.meshgrid(x_shuff[:, 0], t_shuff[:, 0]) 
    
    #Lower and upper bound of the space-time domain
    lb = np.zeros((1,2))
    ub = np.zeros((1,2))

    lb[0, 0] = t_shuff[0,0]
    lb[0, 1] = x_shuff[0,0]
    ub[0, 0] = t_shuff[-1,0]
    ub[0, 1] = x_shuff[-1,0]

    x_norm = standardize(x)
    t_norm = standardize(t)
    y_true_norm = standardize(y_true)
    
    '''
    x_norm = normalize(x)
    t_norm = normalize(t)
    y_true_norm = normalize(y_true)
    '''
    
    X_norm, T_norm = np.meshgrid(x_norm[:, 0], t_norm[:, 0]) 

    # Initial Conditions 
    idx_x0 = np.random.choice(x.shape[0], n_ic, replace=False)

    x0 = x[idx_x0, :]
    X0 = np.column_stack((x0[:,0], np.zeros(len(x0))))

    Y0 = exactSolution(X0[:,1], X0[:,0])
    Y0 = Y0[:, None]

    # Boundary Conditions (normalization builtin)
    idx_t_lb = np.random.choice(t.shape[0], int(n_bc/2), replace=False)
    BC_1_x_t = np.column_stack((x_norm, np.ones(t.shape[0]) * t_norm[0]))[idx_t_lb]
    BC_1_y = y_true_norm[0, :]
    BC_1_y = BC_1_y[idx_t_lb, None] 

    idx_t_ub = np.random.choice(t.shape[0], int(n_bc/2), replace=False)
    BC_2_x_t = np.column_stack((x_norm, np.ones(t.shape[0]) * t_norm[-1]))[idx_t_ub]
    BC_2_y = y_true_norm[-1, :] 
    BC_2_y = BC_2_y[idx_t_ub, None] 

    # Create collocation points with latin hypercube sampling
    X_T_domain = lb + (ub - lb) * lhs(2, n_domain)

    Y_domain = exactSolution(X_T_domain[:,1], X_T_domain[:,0])
    Y_domain = Y_domain[:, None]

    #Normalization of domain and initial conditions
    '''
    X0[:,0] = normalize(X0[:,0])
    Y0 = normalize(Y0)

    X_T_domain[:,1] = normalize(X_T_domain[:,1])
    X_T_domain[:,0] = normalize(X_T_domain[:,0])
    Y_domain = normalize(Y_domain)
    '''
    
    X0[:,0] = standardize(X0[:,0])
    Y0 = standardize(Y0)

    X_T_domain[:,1] = standardize(X_T_domain[:,1])
    X_T_domain[:,0] = standardize(X_T_domain[:,0])
    Y_domain = standardize(Y_domain)
    
    x0 = torch.tensor(X0[:, 0], requires_grad=True).view(-1,1).float().to(device)
    t0 = torch.tensor(X0[:, 1], requires_grad=True).view(-1,1).float().to(device)
    y0 = torch.tensor(Y0).float().to(device)

    x_lb = torch.tensor(BC_1_x_t[:, 0], requires_grad=True).view(-1,1).float().to(device)
    t_lb = torch.tensor(BC_1_x_t[:, 1], requires_grad=True).view(-1,1).float().to(device)
    y_lb = torch.tensor(BC_1_y).float().to(device)

    x_ub = torch.tensor(BC_2_x_t[:, 0], requires_grad=True).view(-1,1).float().to(device)
    t_ub = torch.tensor(BC_2_x_t[:, 1], requires_grad=True).view(-1,1).float().to(device)
    y_ub = torch.tensor(BC_2_y).float().to(device)

    x_domain = torch.tensor(X_T_domain[:, 0], requires_grad=True).view(-1,1).float().to(device)
    t_domain = torch.tensor(X_T_domain[:, 1], requires_grad=True).view(-1,1).float().to(device)
    y_domain = torch.tensor(Y_domain).float().to(device)

    x_norm = torch.tensor(x_norm, requires_grad=True).view(-1,1).float().to(device)
    t_norm = torch.tensor(t_norm, requires_grad=True).view(-1,1).float().to(device)
    y_true_norm = torch.tensor(y_true_norm).float().to(device)

    return x0, t0, y0, x_lb, t_lb, y_lb, x_ub, t_ub, y_ub, x_domain, t_domain, y_domain, x_norm, t_norm, y_true_norm


# In[382]:


class PINN(nn.Module):
    def __init__(self, 
                 layers, losstype, n_batch,
                 t0, x0, y0, 
                 t_lb, x_lb, y_lb, 
                 t_ub, x_ub, y_ub, 
                 t_domain, x_domain, y_domain, x_norm, t_norm, y_norm):
        
        super(PINN, self).__init__() 

        self.activation = nn.Tanh() # ACTIVATION function
        self.layers = layers
        self.losstype = losstype
        self.n_batch = n_batch
        
        #self.T_weight = nn.Tensor.ones(self.n_batch)
        
        #self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]) # layer structure (INCLUDES Glorot-Xavier inizialization)
        
        self.linears = nn.ModuleList()
        
        for i in range(len(layers) - 1): #Create layers
            self.linears.append(nn.Linear(layers[i], layers[i+1]))  # Fully connected layer
            
            # Add dropout layers
            if i < (len(layers) - 2):  # Avoid dropout on the output layer
                self.linears.append(nn.Dropout(0.0000000001)) # p = probability of nullify each element of the tensor

        
        self.s_list = {}
        self.v_list = {}
   
        for i in range(0, len(self.linears), 2): #(modified Glorot-Xavier inizialization + random weight factorization memorization)
            #n = self.linears[i].in_features
            #gx = 1 / np.sqrt(n)
            #self.linears[i].weight.data.normal_(0, gx) 
            #self.linears[i].bias.data.fill_(0)
            
            mean = 1.0
            std = 0.1
        
            w = self.linears[i].weight
            
            # Generate scaling vector s and pivot matrix v for weight factorization
            s = mean + torch.normal(mean, std, size=(w.shape[-1],))
            s = torch.exp(s)
            self.s_list[f"s_{i}"] = nn.Parameter(s, requires_grad=True)
            self.register_parameter(f"s_{i}", self.s_list[f"s_{i}"])  # Register the parameter
            
            v = w / s
            self.v_list[f"v_{i}"] = nn.Parameter(v, requires_grad=True)
            self.register_parameter(f"v_{i}", self.v_list[f"v_{i}"]) # Register the parameter
        
        
        self.t0 = t0
        self.x0 = x0
        self.y0 = y0
        
        self.t_lb = t_lb
        self.x_lb = x_lb
        self.y_lb = y_lb
        
        self.t_ub = t_ub
        self.x_ub = x_ub
        self.y_ub = y_ub
        
        self.t_domain = t_domain
        self.x_domain = x_domain
        self.y_domain = y_domain

        self.x_norm = x_norm
        self.t_norm = t_norm
        self.y_norm = y_norm
        
        # Batches portions too
        self.batch_x0 = 0
        self.batch_t0 = 0
        self.batch_y0 = 0 

        self.batch_x_ub = 0
        self.batch_t_ub = 0
        self.batch_y_ub = 0
        
        self.batch_x_lb = 0 
        self.batch_t_lb = 0
        self.batch_y_lb = 0
        
        self.batch_x_domain = 0
        self.batch_t_domain = 0
        self.batch_y_domain = 0
        
        self.optimizer = []
        self.train_loss_history = []

    
    def get_factorized_weight(self, i):        
        b = self.linears[i].bias

        s = self.s_list[f"s_{i}"]
        v = self.v_list[f"v_{i}"]
        
        return s * v, b
    
    
    def forward(self, X): # Forward pass using decomposed weights with dropout and skip connections
        a = X.float()
        
        for i in range(0, len(self.linears), 2):  # Skip the dropout layers
            
            a_prev = a
            
            kernel, b = self.get_factorized_weight(i)
            a = torch.matmul(a_prev, kernel.T) + b  

            #a = self.linears[i](a_prev)
            
            #Apply activation + dropout only for hidden layers
            if i < (len(self.linears) - 1):  
                a = self.activation(a)
                a = self.linears[i+1](a)
                
                if 0 < i : #Skip connections are activated only after the input layer, included the output
                    if a.shape != a_prev.shape: #In case of layers of different size
                    # Apply a 1x1 linear transformation to match dimensions, but after activation
                        projection = nn.Linear(a_prev.shape[1], a.shape[1], bias=False).to(a.device)
                        a_prev = projection(a_prev) 
                    
                    a += a_prev

        return a
        
    
    def network_prediction(self, t, x):
        
        return self.forward(torch.cat((t, x), 1))

    
    def PDE_prediction(self, t, x): # Compute the differential equation
        N = self.network_prediction(t, x)
        dN_dt = self.get_derivative(N, t, 1)
        #dN_dxx = self.get_derivative(N, x, 2)
        #f = dN_dt - dN_dxx + torch.exp(-t)*(torch.sin(np.pi*x) - np.pi*np.pi*torch.sin(np.pi*x))
        f =  dN_dt - torch.pow(N, 2) - 1 
        
        return f

    
    def get_derivative(self, y, x, n): # General formula to compute the n-th order derivative of y = f(x) with respect to x
        if n == 0:  # (n is the order if the derivative)
            return y
        else:
            dy_dx = torch.autograd.grad(y, x, torch.ones_like(y).to(device), create_graph=True, retain_graph=True, allow_unused=True)[0]
        
        return self.get_derivative(dy_dx, x, n - 1)

    '''
    Creating the additional losses terms given the Lagrange planetary equations
    '''
    def loss_IC(self): 
        y_pred_IC = self.network_prediction(self.batch_t0, self.batch_x0)
        
        if self.losstype == 'mse':
            loss_IC = torch.mean((self.batch_y0 - y_pred_IC) ** 2).to(device)
            
        elif self.losstype == 'logcosh':
            loss_IC = torch.mean(torch.log(torch.cosh(self.batch_y0 - y_pred_IC))).to(device)
    
        return loss_IC     

    
    def loss_BC(self, part):
        if self.losstype == 'mse':
            if part == 0:
                y_pred_BC = self.network_prediction(self.batch_t_lb, self.batch_x_lb)
                loss_BC = torch.mean((self.batch_y_lb - y_pred_BC)**2).to(device)
            else:
                y_pred_BC = self.network_prediction(self.batch_t_ub, self.batch_x_ub)
                loss_BC = torch.mean((self.batch_y_ub - y_pred_BC)**2).to(device)
        
        elif self.losstype == 'logcosh':
           if part == 0:
               y_pred_BC = self.network_prediction(self.batch_t_lb, self.batch_x_lb)
               loss_BC = torch.mean(torch.log(torch.cosh(self.batch_y_lb - y_pred_BC))).to(device)

           else:
               y_pred_BC = self.network_prediction(self.batch_t_ub, self.batch_x_ub)
               loss_BC = torch.mean(torch.log(torch.cosh(self.batch_y_ub - y_pred_BC))).to(device)
        
        return loss_BC
    

    def loss_BC_symmetric(self):
        loss_BC_symmetric = 0.0
        
        if self.losstype == 'mse':
            y_pred = self.network_prediction(self.t_norm, self.x_norm)
            loss_BC_symmetric += torch.mean((y_pred[:,0] - y_pred[:,-1])**2).to(device)
            loss_BC_symmetric += torch.mean((self.y_norm[:,0] - y_pred[:,0])**2).to(device)
            loss_BC_symmetric += torch.mean((self.y_norm[:,-1] - y_pred[:,-1])**2).to(device)
        
        elif self.losstype == 'logcosh':
            y_pred = self.network_prediction(self.t_norm, self.x_norm)
            loss_BC_symmetric += torch.mean(torch.log(torch.cosh(y_pred[:,0] - y_pred[:,-1]))).to(device)
            loss_BC_symmetric += torch.mean(torch.log(torch.cosh(self.y_norm[:,0] - y_pred[:,0]))).to(device)
            loss_BC_symmetric += torch.mean(torch.log(torch.cosh(self.y_norm[:,-1] - y_pred[:,-1]))).to(device)
            
        return loss_BC_symmetric

    
    def loss_interior(self):
        f_pred = self.PDE_prediction(self.batch_t_domain, self.batch_x_domain)
        
        if self.losstype == 'mse':
            loss_interior = torch.mean(torch.pow(f_pred,2)).to(device)
            
        elif self.losstype == 'logcosh':
            loss_interior = torch.mean(torch.log(torch.cosh(f_pred))).to(device)
            
        '''
        eps = 0.7 # T_weight slope: recommended eps = 1

        loss_interior = torch.mean(self.T_weight*torch.log(torch.cosh(f_pred))).to(device)

        for i in range(1, self.n_batch):
            self.T_weight[i] = torch.exp(-eps*torch.sum(loss_interior[1:i]))
        '''        
        return loss_interior

    
    def get_training_history(self):
        loss_his = np.array(self.train_loss_history)
        total_loss, loss_IC, loss_BC, loss_domain, loss_BC_symmetric = np.split(loss_his, 5, axis=1) 
        
        return total_loss, loss_IC, loss_BC, loss_domain, loss_BC_symmetric 


    def loss_func(self):
        loss_IC = self.loss_IC()
        loss_BC = self.loss_BC(0) + self.loss_BC(1) 
        loss_domain = self.loss_interior()
        loss_BC_symmetric = self.loss_BC_symmetric()
        
        return loss_IC, loss_BC, loss_domain, loss_BC_symmetric 

    
    def closuring(self):
        self.optimizer.zero_grad()
        loss_0, loss_b, loss_f, loss_b_sym = self.loss_func() 
        total_loss = loss_0 + loss_b + loss_f + loss_b_sym 
        total_loss.backward(retain_graph=True)
        
        return total_loss

    
    def train_network(self, epochs, optim, batch_size, learning_rate, regularization):
        if optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate, weight_decay = regularization, amsgrad=True)
            
        elif optim == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.parameters(), lr = learning_rate, weight_decay = regularization, amsgrad=True)

        elif optim == 'L-BFGS':
            self.optimizer = torch.optim.LBFGS(self.parameters(), lr = learning_rate, max_iter= 15, history_size=120)

        elif optim == 'SGD':  
            self.optimizer = torch.optim.SGD(self.parameters(), lr = learning_rate, momentum=0.9, weight_decay = regularization, nesterov=True)

        elif optim == 'Adagrad':  
            self.optimizer = torch.optim.Adagrad(self.parameters(), lr = learning_rate, lr_decay = learning_rate*1e-2, weight_decay = regularization)
        
        elif optim == 'ASGD': 
            self.optimizer = torch.optim.ASGD(self.parameters(), lr = learning_rate, alpha = 0.8, weight_decay = regularization, t0 = 1e3)
            
        # Training loop
        for epoch in range(epochs):
            loss_IC, loss_BC, loss_domain, loss_BC_symmetric = 0.0, 0.0, 0.0, 0.0 

            # Mini-batch training for Initial Conditions (IC)
            for self.batch_x0, self.batch_t0, self.batch_y0 in batch_generator(self.x0, self.t0, self.y0, batch_size):
                loss_IC_batch = self.loss_IC()
                loss_IC += loss_IC_batch
            
            # Mini-batch training for Boundary Conditions (BC)
            for self.batch_x_lb, self.batch_t_lb, self.batch_y_lb in batch_generator(self.x_lb,  self.t_lb, self.y_lb, batch_size):
                loss_BC_batch_lb = self.loss_BC(0) # 0 for lower bound
                loss_BC += loss_BC_batch_lb

            for self.batch_x_ub, self.batch_t_ub, self.batch_y_ub in batch_generator(self.x_ub, self.t_ub, self.y_ub, batch_size):
                loss_BC_batch_ub = self.loss_BC(1) # 1 for upper bound
                loss_BC += loss_BC_batch_ub
            
            # Mini-batch training for Domain Loss (interior) and update temporal weights
            for self.batch_x_domain, self.batch_t_domain, self.batch_y_domain in batch_generator(self.x_domain, self.t_domain, self.y_domain, batch_size): 
                loss_domain_batch = self.loss_interior()
                loss_domain += loss_domain_batch

            # Mini-batch training for Simmetric boundary (BC_symmetric)
            loss_BC_symmetric += self.loss_BC_symmetric()

            
            if optim == 'AdamW' or optim == 'Adam':
                # Total loss for this epoch
                total_loss = loss_IC + loss_BC + loss_domain + loss_BC_symmetric  
                self.train_loss_history.append([total_loss.cpu().detach(), loss_IC.cpu().detach(), loss_BC.cpu().detach(), loss_domain.cpu().detach(), loss_BC_symmetric.cpu().detach()]) #, 
                             
                self.optimizer.zero_grad()

                # Calculate gradients
                total_loss.backward()
                
                if epoch % 100 == 0:
                    print(self.s_list[f"s_{0}"].grad)
                    #print(self.linears[0].weight.grad)
                
                # Optimize the network parameters
                self.optimizer.step() 
                
            else:
                # Total loss for this epoch
                total_loss = loss_IC  + loss_BC + loss_domain + loss_BC_symmetric  
                self.train_loss_history.append([total_loss.cpu().detach(), loss_IC.cpu().detach(), loss_BC.cpu().detach(), loss_domain.cpu().detach(), loss_BC_symmetric.cpu().detach()]) #, loss_BC.cpu().detach()

                self.optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                # Optimize the network parameters (with closure)
                self.optimizer.step(self.closuring)
                
            # Print out the loss every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch ({optim}): {epoch}, Total Loss: {total_loss.detach().cpu().numpy()}')


# In[383]:


def exactSolution(t, x):
    
    return np.power(x, 2) + 1
    
x_len = 100
t_len = 100

'''
x = np.random.uniform(-5, 5, x_len)
x = np.sort(x)
x = x.reshape(-1,1)

t = np.random.uniform(0, 5, t_len)
t = np.sort(t)
t = t.reshape(-1,1)
'''

x = np.linspace(-5, 5, x_len).reshape(-1,1) # Space domain (must be same space as time one (for shuffle purposes))
t = np.linspace(0, 5, t_len).reshape(-1,1) # Time domain

X, T = np.meshgrid(x[:, 0], t[:, 0]) 

y_true = exactSolution(T, X) # NOT normalized and NOT reshuffled


# In[384]:


layers = [2, 64, 64, 64, 1] #[2, 8, 16, 8, 32, 8, 16, 8, 1]
losstype = 'mse'
n_batch = 64
epochs = 1000
L_rate = 0.0002
lambda_reg = 0 #0.00002
#frac = 4/5

x0, t0, y0, x_lb, t_lb, y_lb, x_ub, t_ub, y_ub, x_domain, t_domain, y_domain, x_norm, t_norm, y_norm = data_handler(x, t, y_true, 100, 200, x_len*t_len)

model = PINN(layers, losstype, n_batch, t0, x0, y0, t_lb, x_lb, y_lb, t_ub, x_ub, y_ub, t_domain, x_domain, y_domain, x_norm, t_norm, y_norm).to(device)


# In[ ]:


model.train_network(epochs, 'Adam', n_batch, L_rate, lambda_reg) # L-BFGS, AdamW, SGD, ASGD, Adagrad


# In[ ]:


total_loss, loss_IC, loss_BC, loss_domain, loss_BC_symmetric = model.get_training_history() 
#total_loss, loss_IC, loss_BC, loss_domain, loss_prediction = model.get_training_history()

# training/validation losses
plt.figure(figsize=(10, 6))
plt.plot(total_loss, label='Total Loss')
plt.plot(loss_IC, label='Initial Condition Loss')
plt.plot(loss_BC, label='Boundary Condition Loss')
plt.plot(loss_domain, label='Domain Loss')
plt.plot(loss_BC_symmetric, label='Symmetric-boundary Loss')
#plt.plot(loss_prediction, label='Prediction Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


'''
# Get model predictions after training
x_valid = np.linspace(-5, 5, n_valid).reshape(-1,1) # Space domain (must be same space as time one (for shuffle purposes))
t_valid = np.linspace(0, 5, n_valid).reshape(-1,1) # Time domain

X_valid, T_valid = np.meshgrid(x_valid[:, 0], t_valid[:, 0]) 

y_true_valid = exactSolution(T_valid, X_valid) 


x0_val, t0_val, y0_val, x_lb_val, x_lb_val, t_lb_val, y_lb_val, x_ub_val, t_ub_val, y_ub_val, x_domain_val, t_domain_val, y_domain_val = data_handler(x_valid, t_valid, y_true_valid, 50, 80, 1000)

model_valid = PINN(model.layers, model.losstype, t0_val, x0_val, y0_val, t_lb_val, x_lb_val, y_lb_val, t_ub_val, x_ub_val, y_ub_val, t_domain_val, x_domain_val, y_domain_val).to(device)

model_valid.train_network(epochs, 'Adam', n_batch, L_rate, lambda_reg)
'''


# In[ ]:


'''
total_loss_val, loss_IC_val, loss_BC_val, loss_domain_val, loss_BC_symmetric_val = model.get_training_history()
'''


# In[ ]:


# Get model predictions after training
n_valid = 40

'''
x_valid = np.random.uniform(-5, 5, n_valid)
x_valid = np.sort(x_valid)
x_valid = x_valid.reshape(-1,1)

t_valid = np.random.uniform(0, 5, n_valid)
t_valid = np.sort(t_valid)
t_valid = t_valid.reshape(-1,1)
'''

x_valid = np.linspace(-5, 5, n_valid).reshape(-1,1) # Space domain (must be same space as time one (for shuffle purposes))
t_valid = np.linspace(0, 5, n_valid).reshape(-1,1) # Time domain

X_valid, T_valid = np.meshgrid(x_valid[:, 0], t_valid[:, 0]) 

y_true_valid = exactSolution(T_valid, X_valid) 

'''
x_valid = normalize(x_valid)
t_valid = normalize(t_valid)
y_true_valid = normalize(y_true_valid)
'''

x_valid = standardize(x_valid)
t_valid = standardize(t_valid)
y_true_valid = standardize(y_true_valid)

X_valid, T_valid = np.meshgrid(x_valid[:, 0], t_valid[:, 0])                                                                                                  

idx = torch.randperm(n_valid)
#idx_0 = torch.torch.arange(0, 100, 1).int()

x_val = x_valid[idx]
t_val = t_valid[idx]

X_val, T_val = np.meshgrid(x_val[:, 0], t_val[:, 0])

t_pred = torch.tensor(T_val.flatten(), requires_grad=True).view(-1, 1).float().to(device)
x_pred = torch.tensor(X_val.flatten(), requires_grad=True).view(-1, 1).float().to(device)

# Predict with the trained model
y_pred = model.network_prediction(t_pred, x_pred).cpu().detach().numpy()

# Reshape to match the shape of the true values
#y_pred = y_pred.reshape(n_valid, n_valid)



# In[ ]:


'''
plt.figure(figsize=(10, 8))
plt.contourf(X_valid, T_valid, y_true_valid, levels=int(n_valid/2), cmap='jet', alpha=0.7)  # Use alpha for transparency of background
plt.colorbar(label='Exact solution')
plt.contour(X_val, T_val, y_pred, levels=int(n_valid/2), cmap='jet', alpha=0.8)
plt.colorbar(label='Estimated solution')
plt.xlabel('X (Space Domain)')
plt.ylabel('T (Time Domain)')
plt.show()
'''

fig, axes = plt.subplots(1, 2, figsize=(15, 8))

# Subplot 1: Plot true values (exact solution)
c1 = axes[0].contourf(X_valid, T_valid, y_true_valid, levels=int(n_valid/2), cmap='jet', alpha=1)
axes[0].set_xlabel('X')
axes[0].set_ylabel('T')
fig.colorbar(c1, ax=axes[0], label='Exact solution')

# Subplot 2: Plot only predictions (estimated solution)
c2 = axes[1].scatter(X_val, T_val, c=y_pred, cmap='jet', s=n_valid/2, alpha=1)
axes[1].set_xlabel('X')
axes[1].set_ylabel('T')
fig.colorbar(c2, ax=axes[1], label='Estimated solution')
#plt.tight_layout()

plt.show()


# In[ ]:


# Plot Initial conditions and Boundary conditions
plt.figure(figsize=(10, 6))
plt.scatter(x0.detach().numpy(), t0.detach().numpy(), c=y0.flatten(), cmap='jet')
plt.scatter(x_lb.detach().numpy(), t_lb.detach().numpy(), c=y_lb.flatten(), cmap='jet')
plt.scatter(x_ub.detach().numpy(), t_ub.detach().numpy(), c=y_ub.flatten(), cmap='jet')
plt.colorbar(label='Conditions value')
plt.xlabel('Position (x)')
plt.ylabel('Time (t)')
plt.grid(True)
plt.show()

