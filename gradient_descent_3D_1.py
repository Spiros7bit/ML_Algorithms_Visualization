'''
Gradient Descent in 3 Dimensions:
    Visualization the gradient descent with a moving ball 

'''

# Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function
def f(x, y):
    return np.e ** -((x ** 2 + y ** 2) / 2)

# Grad of function respect to x,y return the gradx, grady as list
def grad_f(x, y):
    return np.array([-x * f(x, y), -y * f(x, y)])

# Learning rate, the factor of grad that subtract from x and y
beta = 0.05

# Setup the x and y domain
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

# The meshgrid() function creates a 2D array of x and a 2D array of y
X, Y = np.meshgrid(x, y)

# Create the 3D function
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, f(X, Y), cmap='coolwarm')

# Add labels and features
ax.grid('--')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Define the initial value of x and y
xy_init = np.array([0.5, -0.5, f(0.05, -0.5)])

# Create a plot of the initial position of the ball
ball, = ax.plot([xy_init[0]], [xy_init[1]], [xy_init[2]], color='y', marker='o')

def update_animation(i):
    global xy_init

    # Calculate the gradient
    grad = grad_f(xy_init[0], xy_init[1])

    xy_init[0] -= beta * grad[0]
    xy_init[1] -= beta * grad[1]
    xy_init[2] = f(xy_init[0], xy_init[1])

    # Update the plot data
    ball.set_data([xy_init[0]], [xy_init[1]])
    ball.set_3d_properties([xy_init[2]])

    return ball,

animation = FuncAnimation(fig, update_animation, frames=100, interval=10, blit=True)

plt.show()
