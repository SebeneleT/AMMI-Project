
import sys

print(sys.version)

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# torch.manual_seed(42)

random_seed = 128


class Sin(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)



class NeuralNet(nn.Module):

    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons):
        super(NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function
        self.activation = nn.Tanh()

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        print(self.n_hidden_layers)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

    def forward(self, x):
        # The forward function performs the set of affine and non-linear transformations defining the network
        # (see equation above)
        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        return self.output_layer(x)


def init_xavier(model):
    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            g = nn.init.calculate_gain('tanh')
            torch.nn.init.xavier_uniform_(m.weight, gain=g)
            # torch.nn.init.xavier_normal_(m.weight, gain=g)
            m.bias.data.fill_(0)

    model.apply(init_weights)


def fit(pinns_calss, training_set_x_boundary, training_set_y_boundary, training_set_collocation, num_epochs, optimizer, verbose=True):
    history = list()

    # Loop over epochs
    for epoch in range(num_epochs):
        if verbose: print("################################ ", epoch, " ################################")

        running_loss = list([0])

        # Loop over batches
        for j, ((inp_train_x, u_train_x), (inp_train_y, u_train_y), (inp_train_c, u_train_c)) in enumerate(zip(training_set_x_boundary, training_set_y_boundary, training_set_collocation)):
            def closure():
                optimizer.zero_grad()
                loss = pinns_calss.compute_loss(inp_train_x, u_train_x, inp_train_y, u_train_y, inp_train_c, u_train_c)
                loss.backward()
                running_loss[0] += loss.item()
                return loss

            # Item 3. below
            optimizer.step(closure=closure)

        print('Loss: ', (running_loss[0] / len(training_set_x_boundary)))
        history.append(running_loss[0])

    return history


class Pinns:
    def __init__(self):
        self.domain_extrema = torch.tensor([[0, 1],  # x dimension
                                            [0, 1]])  # y dimension

        self.approximate_solution = NeuralNet(input_dimension=self.domain_extrema.shape[0], output_dimension=1, n_hidden_layers=4, neurons=20)
        torch.manual_seed(12)
        init_xavier(self.approximate_solution)



    # Function returning the training set S_sb corresponding to the x boundary
    def add_x_boundary_points(self, n_boundary):
        x0 = self.domain_extrema[0, 0]
        xL = self.domain_extrema[1, 1]

        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        x_input_boundary = self.soboleng.draw(n_boundary)
        # torch.random.manual_seed(random_seed)
        # input_boundary = torch.rand([n_boundary, 2]).type(torch.FloatTensor)
        x_input_boundary = self.convert(x_input_boundary)

        x_input_boundary_0 = torch.clone(x_input_boundary)
        x_input_boundary_0[:, 1] = torch.full(x_input_boundary_0[:, 1].shape, x0)

        x_input_boundary_L = torch.clone(x_input_boundary)
        x_input_boundary_L[:, 1] = torch.full(x_input_boundary_L[:, 1].shape, xL)

        x_output_boundary_0 = torch.zeros((x_input_boundary.shape[0], 1))
        x_output_boundary_L = torch.zeros((x_input_boundary.shape[0], 1))

        return torch.cat([x_input_boundary_0, x_input_boundary_L], 0), torch.cat([x_output_boundary_0, x_output_boundary_L], 0)


         # Function returning the training set S_sb corresponding to the y boundary
    def add_y_boundary_points(self, n_boundary):
        y0 = self.domain_extrema[0, 0]
        yL = self.domain_extrema[0, 1]

        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        y_input_boundary = self.soboleng.draw(n_boundary)
        # torch.random.manual_seed(random_seed)
        # input_boundary = torch.rand([n_boundary, 2]).type(torch.FloatTensor)
        y_input_boundary = self.convert(y_input_boundary)

        y_input_boundary_0 = torch.clone(y_input_boundary)
        y_input_boundary_0[:, 1] = torch.full(y_input_boundary_0[:, 1].shape, y0)

        y_input_boundary_L = torch.clone(y_input_boundary)
        y_input_boundary_L[:, 1] = torch.full(y_input_boundary_L[:, 1].shape, yL)

        y_output_boundary_0 = torch.zeros((y_input_boundary.shape[0], 1))
        y_output_boundary_L = torch.zeros((y_input_boundary.shape[0], 1))

        return torch.cat([y_input_boundary_0, y_input_boundary_L], 0), torch.cat([y_output_boundary_0, y_output_boundary_L], 0)   

    # Function returning the training set S_int corresponding to the interior domain where the PDE is enforced
    def add_collocation_points(self, n_collocation):
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        input_collocation = self.soboleng.draw(n_collocation)
        # torch.random.manual_seed(random_seed)
        # input_collocation = torch.rand([n_collocation, 2]).type(torch.FloatTensor)
        input_collocation = self.convert(input_collocation)

        output_collocation = torch.zeros((input_collocation.shape[0], 1))
        return input_collocation, output_collocation


    # Function to compute the terms required in the definition of the x boundary residual
    def apply_x_boundary_conditions(self, x_input_boundary, x_output_boundary):
        pred_x_output_boundary = self.approximate_solution(x_input_boundary)
        assert (pred_x_output_boundary.shape[1] == x_output_boundary.shape[1])
        return x_output_boundary, pred_x_output_boundary

    # Function to compute the terms required in the definition of the y boundary residual
    def apply_y_boundary_conditions(self, y_input_boundary, y_output_boundary):
        pred_y_output_boundary = self.approximate_solution(y_input_boundary)
        assert (pred_y_output_boundary.shape[1] == y_output_boundary.shape[1])
        return y_output_boundary, pred_y_output_boundary    

    # Function to compute the PDE residuals
    def compute_pde_residual(self, input_collocation):
        input_collocation.requires_grad = True
        u = self.approximate_solution(input_collocation).reshape(-1, )
        

        # grad compute the gradient of a "SCALAR" function L with respect to some input nxm TENSOR Z=[[x1, y1],[x2,y2],[x3,y3],...,[xm,yn]], m=2
        # it returns grad_L = [[dL/dx1, dL/dy1],[dL/dx2, dL/dy2],[dL/dx3, dL/dy3],...,[dL/dxm, dL/dyn]]
        # Note: pytorch considers a tensor [u1, u2,u3, ... ,un] a vectorial function
        # whereas sum_u = u1 + u2 u3 + u4 + ... + un as a "scalar" one

        # In our case ui = u(xi), therefore the line below returns:
        # grad_u = [[dsum_u/dx1, dsum_u/dy1],[dsum_u/dx2, dL/dy2],[dsum_u/dx3, dL/dy3],...,[dsum_u/dxm, dL/dyn]]
        # and dsum_u/dxi = d(u1 + u2 u3 + u4 + ... + un)/dxi = d(u(x1) + u(x2) u3(x3) + u4(x4) + ... + u(xn))/dxi = dui/dxi
        grad_u = torch.autograd.grad(u.sum(), input_collocation, create_graph=True)[0]
        grad_u_x = grad_u[:, 0]
        grad_u_y = grad_u[:, 1]
        grad_u_xx = torch.autograd.grad(grad_u_x.sum(), input_collocation, create_graph=True)[0][:, 0]
        grad_u_yy = torch.autograd.grad(grad_u_y.sum(), input_collocation, create_graph=True)[0][:, 1]

        # torch.autograd.grad(grad_u_t.sum(), input_collocation, create_graph=True)[0][:, 1] = torch.autograd.grad(grad_u_x.sum(), input_collocation, create_graph=True)[0][:, 0]

        # u_tt = a**2 u_xx

        #
        ut = torch.sin(2 * np.pi*input_collocation[:, 0]) * torch.sin(2 * np.pi*input_collocation[:, 1])
        grad_ut = torch.autograd.grad(ut.sum(), input_collocation, create_graph=True)[0]
        grad_ut_x = grad_ut[:, 0]
        grad_ut_y = grad_ut[:, 1]
        grad_ut_xx = torch.autograd.grad(grad_ut_x.sum(), input_collocation, create_graph=True)[0][:, 0]
        grad_ut_yy = torch.autograd.grad(grad_ut_y.sum(), input_collocation, create_graph=True)[0][:, 1]

        f = grad_ut_xx + grad_ut_yy

    

        residual = -grad_u_xx - grad_u_yy - f
        return residual.reshape(-1, )

    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])

        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    def compute_loss(self, inp_train_x, u_train_x, inp_train_y, u_train_y, inp_train_c, u_train_c):
        u_train_x, u_pred_x = self.apply_x_boundary_conditions(inp_train_x, u_train_x)
        u_train_y, u_pred_y = self.apply_y_boundary_conditions(inp_train_y, u_train_y)

        r_int = self.compute_pde_residual(inp_train_c)  # - u_train_c.reshape(-1,)
        r_sbx = u_train_x - u_pred_x
        r_sby = u_train_y - u_pred_y

        loss_sbx = torch.mean(abs(r_sbx) ** 2)
        loss_sby = torch.mean(abs(r_sby) ** 2)
        loss_int = torch.mean(abs(r_int) ** 2)

        lambda_u = 0.1

        loss_u = loss_sbx + loss_sby
        loss = torch.log10(lambda_u * (loss_sbx + loss_sby) + loss_int)
        print("Total loss: ", round(loss.item(), 4), "| PDE Loss: ", round(torch.log10(loss_u).item(), 4), "| Function Loss: ", round(torch.log10(loss_int).item(), 4))

        return loss

    def plotting(self):
        inputs = self.soboleng.draw(10000)
        inputs = self.convert(inputs)

        output = self.approximate_solution(inputs)

        plt.scatter(inputs[:, 1].detach().numpy(), inputs[:, 0].detach().numpy(), c=output.detach().numpy().T)
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")

        plt.show()


# Solve the heat equation:
# u_t = k Dx, (t,x) in [0, 0.01]x[-1,1], k=1
# with zero dirichlet BC and
# u(x,0)= -sin(pi x)

#n_init = 1024
n_bound = 32768
n_coll = 65536

pinn = Pinns()
# Generate S_sb, S_tb, S_int
input_x_, output_x_ = pinn.add_x_boundary_points(n_bound)  # S_sb_x
input_y_, output_y_ = pinn.add_y_boundary_points(n_bound)  # S_sb_y
input_c_, output_c_ = pinn.add_collocation_points(n_coll)  # S_int

plt.figure(figsize=(16, 8))
plt.scatter(input_x_[:, 1].detach().numpy(), input_x_[:, 0].detach().numpy(), label="X Boundary Points")
plt.scatter(input_y_[:, 0].detach().numpy(), input_y_[:, 1].detach().numpy(), label="Y Boundary Points")
plt.scatter(input_c_[:, 1].detach().numpy(), input_c_[:, 0].detach().numpy(), label="Interior Points")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

training_set_x = DataLoader(torch.utils.data.TensorDataset(input_x_, output_x_), batch_size=2 * n_bound, shuffle=False)
training_set_y = DataLoader(torch.utils.data.TensorDataset(input_y_, output_y_), batch_size=2 * n_bound, shuffle=False)
training_set_c = DataLoader(torch.utils.data.TensorDataset(input_c_, output_c_), batch_size=n_coll, shuffle=False)

n_epochs = 1
optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(), lr=float(0.5), max_iter=1000, max_eval=50000, history_size=150,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1.0 * np.finfo(float).eps)
hist = fit(pinn, training_set_x, training_set_y, training_set_c, num_epochs=n_epochs, optimizer=optimizer_LBFGS, verbose=True)

pinn.plotting()
