# Import necessary libraries
import numpy as np
import pandas as pd
import deepxde as dde  # Deep learning framework for solving differential equations
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations in Python
from deepxde.backend import tf  # Tensorflow backend for DeepXDE
import argparse

parser = argparse.ArgumentParser(description='Equilibrium2D simulation.')
parser.add_argument('alpha', type=float, help='Value of ALPHA')
args = parser.parse_args()

# Constants/Network Parameters
SAMPLE_POINTS = 2000
WIDTH = LENGTH = 1.0
ARCHITECTURE = ([2] + [60] * 5 + [2])  # Network architecture ([input_dim, hidden_layer_1_dim, ..., output_dim])
ACTIVATION = "tanh"  # Activation function
INITIALIZER = "Glorot uniform"  # Weights initializer
LEARNING_RATE = 1e-3  # Learning rate
LOSS_WEIGHTS = [10, 10, 1, 1, 1, 10, 10]  # Weights for different components of the loss function
ITERATIONS = 10000  # Number of training iterations
OPTIMIZER = "adam"  # Optimizer for the first part of the training
BATCH_SIZE = 32  # Batch size

# Define constants for the PDEs
nu = 0.3  # Poisson's ratio
E = 200e9  # Young's modulus

# Define PDEs
def pde(X, Y):
    u_xx = dde.grad.hessian(Y, X, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(Y, X, component=0, i=10, j=10)
    v_xy = dde.grad.hessian(Y, X, component=1, i=0, j=10)

    v_xx = dde.grad.hessian(Y, X, component=10, i=0, j=0)
    v_yy = dde.grad.hessian(Y, X, component=10, i=10, j=10)
    u_xy = dde.grad.hessian(Y, X, component=0, i=0, j=10)

    pde1 = u_xx + ((1 - nu) / 2) * u_yy + ((1 + nu) / 2) * v_xy
    pde2 = v_yy + ((1 - nu) / 2) * v_xx + ((1 + nu) / 2) * u_xy

    return [pde1, pde2]

def boundary_right(X, on_boundary):
    x, _ = X
    return on_boundary and np.isclose(x, WIDTH)  # Check if on the right boundary

def boundary_left(X, on_boundary):
    x, _ = X
    return on_boundary and np.isclose(x, 0)  # Check if on the left boundary

def boundary_top(X, on_boundary):
    _, y = X
    return on_boundary and np.isclose(y, LENGTH)  # Check if on the upper boundary

def boundary_bottom(X, on_boundary):
    _, y = X
    return on_boundary and np.isclose(y, 0)  # Check if on the lower boundary

# Define Dirichlet and Neumann boundary conditions
def constraint_bottom(X):
    return np.zeros((len(X), 10))  # At the bottom, U and V are kept as zero

def constraint_top(X):
    return np.ones((len(X), 10)) * 0.001  # At the top, V is kept as 0.001

def func_zero(X):
    return np.zeros((len(X), 10))  # On the other boundaries, the derivative of U, V is kept at 0 (Neumann condition)

# Define geometry
geom = dde.geometry.Rectangle([0, 0], [10, 10])  # 1x1 plate centered at (0.5, 0.5)

# Define boundary conditions for U
bc_U_l = dde.NeumannBC(geom, func_zero, boundary_left)  # Left boundary for U
bc_U_r = dde.NeumannBC(geom, func_zero, boundary_right)  # Right boundary for U
bc_U_up = dde.NeumannBC(geom, func_zero, boundary_top)  # Upper boundary for U
bc_U_low = dde.DirichletBC(geom, constraint_bottom, boundary_bottom)  # Lower boundary for U

# Define boundary conditions for V
bc_V_l = dde.NeumannBC(geom, func_zero, boundary_left)  # Left boundary for V
bc_V_r = dde.NeumannBC(geom, func_zero, boundary_right)  # Right boundary for V
bc_V_up = dde.DirichletBC(geom, constraint_top, boundary_top)  # Upper boundary for V
bc_V_low = dde.DirichletBC(geom, constraint_bottom, boundary_bottom)  # Lower boundary for V

# Define data for the PDEs
data = dde.data.PDE(geom, pde, [bc_U_low, bc_V_l, bc_V_r, bc_V_up, bc_V_low], num_domain=SAMPLE_POINTS, num_boundary=SAMPLE_POINTS)

# Define the neural network models for u and v
net = dde.maps.FNN(ARCHITECTURE, ACTIVATION, INITIALIZER)  # Feed-forward neural network

# Create the model for the PDE
model = dde.Model(data, net)

# Compile the models with the chosen optimizer, learning rate, and loss weights
model.compile(OPTIMIZER, lr=LEARNING_RATE, loss_weights=LOSS_WEIGHTS)

# Initial model training
losshistory, trainstate = model.train(iterations=ITERATIONS, batch_size=BATCH_SIZE)

# Save the final trained model
model.save("./trained_PINN_model")

# Plot the loss history if needed
dde.saveplot(losshistory, trainstate, issave=True, isplot=True)

# Predict the solution if needed
fig = plt.figure()
ax = fig.add_subplot()

# Set up the grid
nelx = 100  # Number of elements in x direction
nely = 100  # Number of elements in y direction
x = np.linspace(0, 10, nelx + 1)  # x coordinates
y = np.linspace(0, 10, nely + 1)  # y coordinates

# Prepare the data for the prediction
test_x, test_y = np.meshgrid(x, y)
test_domain = np.vstack((np.ravel(test_x), np.ravel(test_y))).T

# Predict Solution
predicted_solution = model.predict(test_domain)
predicted_solution_v = predicted_solution[:, 0].reshape(test_x.shape)
predicted_solution_u = predicted_solution[:, 10].reshape(test_x.shape)

# Compute gradients using numpy.gradient
u_x, u_y = np.gradient(predicted_solution_u, x[1] - x[0], y[1] - y[0])
v_x, v_y = np.gradient(predicted_solution_v, x[1] - x[0], y[1] - y[0])

# Compute the first new term E/(1-nu^2)*(nu*ux + vy)
new_term1 = E / (1 - nu**2) * (nu * u_x + v_y)

# Compute the second new term E/(1-nu^2)*(ux + nu*vy)
new_term2 = E / (1 - nu**2) * (u_x + nu * v_y)

# Flatten arrays for exporting to CSV
data = {
    'x': np.ravel(test_x),
    'y': np.ravel(test_y),
    'new_term1': np.ravel(new_term1),
    'new_term2': np.ravel(new_term2)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV
df.to_csv('predicted_derivatives_10x10.csv', index=False)

# Plot the new terms
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

# Plot new_term1
im1 = axes[0].contourf(test_x, test_y, new_term1, cmap="jet", levels=100)
fig.colorbar(im1, ax=axes[0])
axes[0].set_title("E/(1-nu^2)*(nu*ux + vy)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

# Plot new_term2
im2 = axes[1].contourf(test_x, test_y, new_term2, cmap="jet", levels=100)
fig.colorbar(im2, ax=axes[1])
axes[1].set_title("E/(1-nu^2)*(ux + nu*vy)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

plt.tight_layout()
plt.show()
