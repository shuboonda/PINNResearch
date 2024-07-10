# Import necessary libraries
import deepxde as dde  # Deep learning framework for solving differential equations
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations in Python
import numpy as np  # For numerical operations
from deepxde.backend import tf  # Tensorflow backend for DeepXDE
import matplotlib.animation as animation  # For creating animations
from matplotlib.animation import FuncAnimation  # Function-based interface to create animations
import argparse

parser = argparse.ArgumentParser(description='Equilibrium2D simulation.')
parser.add_argument('alpha', type=float, help='Value of ALPHA')
args = parser.parse_args()

# Constants/Network Parameters
SAMPLE_POINTS = 2000
WIDTH = LENGTH = 1.0
ARCHITECTURE = (
    [2, [40, 40], [40, 40], [40, 40], 2]
)  # Network architecture ([input_dim, hidden_layer_1_dim, ..., output_dim])
ACTIVATION = "tanh"  # Activation function
INITIALIZER = "Glorot uniform"  # Weights initializer
LEARNING_RATE = 1e-3  # Learning rate
LOSS_WEIGHTS = [
    10,
    10,
    1,
    1,
    1,
    10,
    10,
]  # Weights for different components of the loss function
ITERATIONS = 10000  # Number of training iterations
OPTIMIZER = "adam"  # Optimizer for the first part of the training
BATCH_SIZE = 32  # Batch size

# Define constants for the PDEs
nu = 0.3  # Poisson's ratio

# Define PDEs
def pde(X, Y):
    u_xx = dde.grad.hessian(Y, X, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(Y, X, component=0, i=1, j=1)
    v_xy = dde.grad.hessian(Y, X, component=1, i=0, j=1)

    v_xx = dde.grad.hessian(Y, X, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(Y, X, component=1, i=1, j=1)
    u_xy = dde.grad.hessian(Y, X, component=0, i=0, j=1)

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
    return np.zeros((len(X), 1))  # At the bottom, U and V are kept as zero

def constraint_top(X):
    return (np.ones((len(X), 1)) * 0.001)  # At the top, V is kept as 0.001

def func_zero(X):
    return np.zeros((len(X), 1))  # On the other boundaries, the derivative of U, V is kept at 0 (Neumann condition)

# Define geometry
geom = dde.geometry.Rectangle([0, 0], [1, 1])  # 1x1 plate centered at (0.5, 0.5)

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
net = dde.maps.PFNN(ARCHITECTURE, ACTIVATION, INITIALIZER)  # Feed-forward neural network

# Create the model for the PDE
model = dde.Model(data, net)

# Compile the models with the chosen optimizer, learning rate, and loss weights
model.compile(OPTIMIZER, lr=LEARNING_RATE, loss_weights=LOSS_WEIGHTS)

# Initial model training
losshistory, trainstate = model.train(iterations=ITERATIONS, batch_size=BATCH_SIZE)

# Save the final trained model
model.save("./trained_PINN_model")

# Train the models
losshistory, trainstate = model.train(iterations=ITERATIONS, batch_size=BATCH_SIZE)

# Save the models if needed
model.save("./trained_PINN_model")

# Plot the loss history if needed
dde.saveplot(losshistory, trainstate, issave=True, isplot=True)

## TODO - ALL LINES UNDER THIS ##

# Predict the solution if needed
fig = plt.figure()
ax = fig.add_subplot()

# Set up the grid
nelx = 100  # Number of elements in x direction
nely = 100  # Number of elements in y direction
x = np.linspace(0, 1, nelx + 1)  # x coordinates
y = np.linspace(0, 1, nely + 1)  # y coordinates

# Prepare the data for the prediction
test_x, test_y = np.meshgrid(x, y)
test_domain = np.vstack((np.ravel(test_x), np.ravel(test_y))).T

# Predict Solution
predicted_solution = model.predict(test_domain)
predicted_solution_u = predicted_solution[:, 1]
predicted_solution_u = predicted_solution_u.reshape(test_x.shape)
predicted_solution_v = predicted_solution[:, 0]
predicted_solution_v = predicted_solution_v.reshape(test_x.shape)

# Predict Residual
residual = model.predict(test_domain, operator=pde)

# Extract and print final values of u_x, u_y, v_x, v_y at a specific point (e.g., center of the domain)
def gradient_u(x, y):
    return dde.grad.jacobian(predicted_solution, test_domain, i=0)

def gradient_v(x, y):
    return dde.grad.jacobian(predicted_solution, test_domain, i=1)

center = np.array([[0.5, 0.5]])
u_x, u_y = gradient_u(center[0, 0], center[0, 1])
v_x, v_y = gradient_v(center[0, 0], center[0, 1])

print(f"u_x at center: {u_x}")
print(f"u_y at center: {u_y}")
print(f"v_x at center: {v_x}")
print(f"v_y at center: {v_y}")

# Plot the combined solution
plt.contourf(test_x, test_y, predicted_solution_u, cmap="jet", levels=100)
plt.colorbar()
plt.title(f"Surface: Dependant Variable U")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

plt.contourf(test_x, test_y, predicted_solution_v, cmap="jet", levels=100)
plt.colorbar()
plt.title(f"Surface: Dependant Variable V")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
