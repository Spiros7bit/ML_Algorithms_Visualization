'''
Gradient Descent in 2 Dimensions:
    Visualization the gradient descent with a moving ball 

'''
# Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function
def f(x):
    return x**3 - 3*x**2

# Grad of function
def grad_f(x):
    return 3*x**2 - 6*x

# Learning rate, the factor of grad that substract from x 
beta = 0.05 #float( input(' Give beta: ') )

# Initialize the plot
fig = plt.figure()
ax = plt.axes()
ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')
ax.set_xlim(-1, 4)
ax.set_ylim(-10, 25)
plt.grid('--')

# Define the initial value of x and y
x_init = 0.005 #float( input(' Give initial point of ball: ') )
y_init = f(x_init)

# Plot the functions
x = np.linspace(-1, 4, 100)
y = f(x)
dy = grad_f(x)
plt.plot(x, y, label='y')
plt.plot(x, dy, label='dy/dx')

# Plot the ball, and the initial point of ball
ball, = plt.plot([x_init], [y_init], 'go', markersize=15, label='ball') # or with scatter() function

# Initialize the update function for the animation 
def update_animation(j):
    global x_init, y_init
    
    # Update the current point
    x_init -= beta * grad_f(x_init)
    y_init = f(x_init)
    
    # set_data() update the position of the ball
    ball.set_data(x_init, y_init)
    
    return ball, #each time that this function called from FuncAnimation

# Create the animation with, FuncAnimation that call update_animation for each frame 
animation = FuncAnimation(fig, update_animation, frames=1000, blit=True)

# Show the animation and labels
plt.legend()
plt.show()

'''
# Import libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

#user input 
beta = 0.005 #float( input('Beta: ') )
x_root = 3.0

# Function
def function(x):
    return x ** 3 - 3 * x ** 2

# Grad of function
def gradient(x):
    return 3 * x ** 2 - 6 * x 

'''
'''
# Creating vectors X and Y
x = np.arange(-3, 6, 0.001)
y = x ** 3 - 3 * x ** 2
dy = 3 * x ** 2 - 6 * x 

fig = plt.figure(figsize = (12, 7))

# Animation
ax = plt.axes(xlim=(-1, 5), ylim=(-5, 25))
patch = plt.Circle((0, 0), 0.2, fc='g')

def init():
    patch.center = (0.6, 0.6)
    ax.add_patch(patch)
    return patch,

def animate(x_root):
    x_root, y_root = patch.center
    while( ( np.absolute(3 * x_root ** 2 - 6 * x_root ) > 10 ** (-3) ).any() ):
        x_root = x_root - beta * (3 * x_root ** 2 - 6 * x_root )
        y_root = x_root ** 3 - 3 * x_root ** 2
        patch.center = (x_root, y_root)
        plt.pause(0.5)
    return patch,

anim = FuncAnimation(fig, animate, 
                    init_func=init, 
                    frames=50, 
                    interval=500,
                    blit=True)


# Create the plot
plt.title('Gradient Descent f(x) = x^3-3x^2')
plt.plot(x, y, 'b', label = 'y')
plt.plot(x, dy, 'r', label = 'dy/dx')


# Add features to our figure
plt.legend()
plt.grid(True, linestyle =':')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.xlim([-1, 4])
plt.ylim([-5, 5])


# Show the plot
plt.show()


'''