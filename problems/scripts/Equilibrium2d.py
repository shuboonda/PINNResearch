# Import necessary libraries
import deepxde as dde  # Deep learning framework for solving differential equations
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations in Python
import numpy as np  # For numerical operations
from deepxde.backend import tf  # Tensorflow backend for DeepXDE
import matplotlib.animation as animation  # For creating animations
from matplotlib.animation import (
    FuncAnimation,
)  # Function-based interface to create animations
import argparse

parser = argparse.ArgumentParser(description='Equilibrium2D simulation.')
parser.add_argument('alpha', type=float, help='Value of ALPHA')
args = parser.parse_args()

# Constants/Network Parameters
SAMPLE_POINTS = 2000
WIDTH = LENGTH = 1.0
ARCHITECTURE = (
    [2] + [60] * 5 + [1]
)  # Network architecture ([input_dim, hidden_layer_1_dim, ..., output_dim])
ACTIVATION = "tanh"  # Activation function
INITIALIZER = "Glorot uniform"  # Weights initializer
LEARNING_RATE = 1e-3  # Learning rate
LOSS_WEIGHTS = [
    10,
    1,
    1,
    1,
    1,
    10,
]  # Weights for different components of the loss function
ITERATIONS = 10000  # Number of training iterations
OPTIMIZER = "adam"  # Optimizer for the first part of the training
BATCH_SIZE = 32  # Batch size

# Define constants for the PDEs
## TODO - Confirm E and G are no longer needed

E = 200e9  # Elastic modulus
G = 77e9  # Shear modulus
nu = 0.3  # Poisson's ratio

# Define PDEs
def pde1(X, u, v):
    x, y = X[:, 0:1], X[:, 1:2]
    u_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_y = dde.grad.jacobian(u, y, i=0, j=0)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    u_yy = dde.grad.hessian(u, y, i=0, j=0)
    v_xx = dde.grad.hessian(v, x, i=0, j=0)
    v_xy = dde.grad.hessian(v, y, i=0, j=1)

    return (
        u_xx + ((1 - nu) / 2) * u_yy + ((1 - nu) / 2) * v_xy
    )  # First PDE


def pde2(X, v, u):
    x, y = X[:, 0:1], X[:, 1:2]
    v_x = dde.grad.jacobian(v, x, i=0, j=0)
    v_y = dde.grad.jacobian(v, y, i=0, j=0)
    v_xx = dde.grad.hessian(v, x, i=0, j=0)
    v_yy = dde.grad.hessian(v, y, i=0, j=0)
    u_xy = dde.grad.hessian(u, x, i=0, j=1)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)

    return (
        v_yy + ((1 - nu) / 2) * v_xx + ((1 - nu) / 2) * u_xy
    )  # Second PDE

# Define Dirichlet and Neumann boundary conditions

# Define boundary conditions for u
def boundary_conditions_u(X, on_boundary):
    x, y = X[:, 0], X[:, 1]
    return np.where(
        on_boundary,
        # Specify boundary conditions here
        # For example, Dirichlet boundary condition u(x, y) = 0 at all boundaries
        0,
        None,
    )

# Define boundary conditions for v
def boundary_conditions_v(X, on_boundary):
    x, y = X[:, 0], X[:, 1]
    return np.where(
        on_boundary,
        # Specify boundary conditions here
        # For example, Neumann boundary condition ∂v/∂n = 0 (where n is the outward normal vector) at all boundaries
        0,
        None,
    )

# Define geometry
geom = dde.geometry.Rectangle([0, 0], [1, 1])  # 1x1 plate centered at (0.5, 0.5)

# Define boundary conditions for u and v
bc_u = dde.DirichletBC(geom, boundary_conditions_u)
bc_v = dde.DirichletBC(geom, boundary_conditions_v)

# Define data for the PDEs
data = dde.data.PDE(geom, pde1, [bc_u, bc_v], num_domain=SAMPLE_POINTS, num_boundary=SAMPLE_POINTS)

# Define the neural network models for u and v
net_u = dde.maps.FNN(ARCHITECTURE, ACTIVATION, INITIALIZER)  # Feed-forward neural network for u
net_v = dde.maps.FNN(ARCHITECTURE, ACTIVATION, INITIALIZER)  # Feed-forward neural network for v

# Create the model for the first PDE
model_u = dde.Model(data, net_u)

# Create the model for the second PDE
model_v = dde.Model(data, net_v)

# Compile the models with the chosen optimizer, learning rate, and loss weights
model_u.compile(OPTIMIZER, lr=LEARNING_RATE, loss_weights=LOSS_WEIGHTS)
model_v.compile(OPTIMIZER, lr=LEARNING_RATE, loss_weights=LOSS_WEIGHTS)

# Initial model training
losshistory_u, trainstate_u = model_u.train(iterations=ITERATIONS, batch_size=BATCH_SIZE)
losshistory_v, trainstate_v = model_v.train(iterations=ITERATIONS, batch_size=BATCH_SIZE)

# Residual Adaptive Refinement (RAR) for both models
X = geom.random_points(1000)
err_u = 1
err_v = 1
while err_u > 0.01 or err_v > 0.01:
    # RAR for u model
    f_u = model_u.predict(X, operator=pde1)
    err_u = np.mean(np.abs(f_u))
    print("Mean residual (u): %.3e" % (err_u))
    x_id_u = np.argmax(np.abs(f_u))
    print("Adding new point for u:", X[x_id_u], "\n")
    data.add_anchors(X[x_id_u])
    early_stopping_u = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
    model_u.compile(OPTIMIZER, lr=LEARNING_RATE, loss_weights=LOSS_WEIGHTS)
    model_u.train(
        iterations=100,
        disregard_previous_best=True,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping_u],
    )

    # RAR for v model
    f_v = model_v.predict(X, operator=pde2)
    err_v = np.mean(np.abs(f_v))
    print("Mean residual (v): %.3e" % (err_v))
    x_id_v = np.argmax(np.abs(f_v))
    print("Adding new point for v:", X[x_id_v], "\n")
    data.add_anchors(X[x_id_v])
    early_stopping_v = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
    model_v.compile(OPTIMIZER, lr=LEARNING_RATE, loss_weights=LOSS_WEIGHTS)
    model_v.train(
        iterations=100,
        disregard_previous_best=True,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping_v],
    )

# Save the final trained models
model_u.save("./trained_PINN_model_u")
model_v.save("./trained_PINN_model_v")

# Train the models
losshistory_u, trainstate_u = model_u.train(iterations=ITERATIONS, batch_size=BATCH_SIZE)
losshistory_v, trainstate_v = model_v.train(iterations=ITERATIONS, batch_size=BATCH_SIZE)

# Save the models if needed
model_u.save("./trained_PINN_model_u")
model_v.save("./trained_PINN_model_v")

# Plot the loss history if needed
dde.saveplot(losshistory_u, trainstate_u, issave=True, isplot=True)
dde.saveplot(losshistory_v, trainstate_v, issave=True, isplot=True)


## TODO - ALL LINES UNDER THIS ##

# Predict the solution if needed
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set up the grid
nelx = 100  # Number of elements in x direction
nely = 100  # Number of elements in y direction
timesteps = 101  # Number of time steps
x = np.linspace(0, 1, nelx + 1)  # x coordinates
y = np.linspace(0, 1, nely + 1)  # y coordinates
t = np.linspace(0, 1, timesteps)  # Time points

# Prepare the data for the prediction
test_x, test_y, test_t = np.meshgrid(x, y, t)
test_domain = np.vstack((np.ravel(test_x), np.ravel(test_y), np.ravel(test_t))).T

# Predict Solution
predicted_solution_u = model_u.predict(test_domain)
predicted_solution_u = predicted_solution_u.reshape(test_x.shape)  # Scale and reshape solution
predicted_solution_v = model_v.predict(test_domain)
predicted_solution_v = predicted_solution_v.reshape(test_x.shape)  # Scale and reshape solution

# Predict Residual
residual_u = model_u.predict(test_domain, operator=pde1)
residual_u = residual_u.reshape(test_x.shape)  # Reshape residuals
residual_v = model_v.predict(test_domain, operator=pde2)
residual_v = residual_v.reshape(test_x.shape)  # Reshape residuals

# Plot the combined solution
surf1 = ax.plot_surface(
    test_x[:, :, 0],
    test_y[:, :, 0],
    predicted_solution_u[:, :, 0],
    cmap='coolwarm',
    edgecolor='none',
    alpha=0.8,
)
surf2 = ax.plot_surface(
    test_x[:, :, 0],
    test_y[:, :, 0],
    predicted_solution_v[:, :, 0],
    cmap='coolwarm',
    edgecolor='none',
    alpha=0.8,
)
surf3 = ax.plot_surface(
    test_x[:, :, 0],
    test_y[:, :, 0],
    residual_u[:, :, 0],
    cmap='viridis',
    edgecolor='none',
    alpha=0.5,
)
surf4 = ax.plot_surface(
    test_x[:, :, 0],
    test_y[:, :, 0],
    residual_v[:, :, 0],
    cmap='viridis',
    edgecolor='none',
    alpha=0.5,
)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Displacement/Residual')
ax.set_title('Combined Displacement and Residual (u and v)')

# Add colorbars
fig.colorbar(surf1, ax=ax, label='Displacement (u)')
fig.colorbar(surf2, ax=ax, label='Displacement (v)')
fig.colorbar(surf3, ax=ax, label='Residual (u)')
fig.colorbar(surf4, ax=ax, label='Residual (v)')

# Show the plot
plt.show()

