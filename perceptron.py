'''
Name: Spyridon Tsioupros 
email: tsiouprosspiros@gmail.com

Prceptron Algorithm 

'''

# Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Number of protorypes (inputs)
n = 32

# Setup the class 0
x0 = np.random.uniform(0.0, 0.6, size=(n//2, 2) )
# Setup the class 1
x1 = np.random.uniform(0.5, 0.9, size=(n//2, 2) )
# Add bias x = -1 
bias = -1
#second way x0_b = [np.insert(x0[i,:], 0, bias) for i in range(n//2)] 
x0_b = np.insert(x0, 0, bias, axis=1) # Insert in 0 col the bias, axis=1 refers to the columns, if axis=0 all elements of 0 row index will be -1
x1_b = np.insert(x1, 0, bias, axis=1) # Insert in 0 col the bias, axis=1 refers to the columns

# Initialize the weights with random numbers from -10 to 10 (Synapse) with same size as x
w = np.random.uniform(-3, 3, size=(3, 1))

# Initialize the target d with step function values
d = np.concatenate((np.zeros((n//2, 1)), np.ones((n//2, 1))), axis=0)

# Initialize the output y with zeros and if u>0 will be 1
y = np.zeros((n, 1))

x_b = np.concatenate((x0_b, x1_b), axis=0) # Array with all protorypes (inputs + biases)
#print(x_b) #debug
epoch = 0 #counter for epochs
flag = True # flag for error 

################
#### Plots #####
################
# Setup the (2, 2) plots in 1 figure
fig, axs = plt.subplots(2, 2, figsize=(8,8))

ax = axs[0, 0] # set the variable ax = ax[0, 0] the first plot
##### 1st plot with data
ax.grid('--')
ax.plot(x0[:,0], x0[:,1], 'ro', label='Class A') # x all from col 0, for y all from col 1
ax.plot(x1[:,0], x1[:,1], 'gv', label='Class B') # x all from col 0, for y all from col 1
ax.set_title('Inputs')
ax.legend

ax = axs[0, 1] # set the variable ax = ax[0, 1] the second plot
##### 2nd plot with moving line 
ax.grid('--')
ax.plot(x_b[:,1], x_b[:,2],'bo')
ax.set_title('Train')

ax = axs[1, 0] # set the variable ax = ax[1, 0] the third plot
##### 3nd plot with outputs points 
ax.grid('--')
ax.set_title('Output')
ax.legend
ax.set_ylim(-0.5,1.1)
ax.set_xlim(-0.1,1.1)

ax = axs[1, 1]# set the variable ax = ax[1, 1] the fourth plot
##### 4nd plot with outputs points 
ax.grid('--')
ax.set_title('Error')
ax.set_ylim(-1.1, 1.1)
ax.set_xlim(-1.1, 1000)


################
####Training####
################
# Epochs, number of loops until training ends
max_epochs = 1000000
beta = 0.01 

line, = axs[0, 1].plot([], [], 'r-') # 
output_u, = axs[1, 0].plot([], [], 'ro')
error, = axs[1, 1].plot([], [], 'r-')

k = [] #This list is a counter of variable i, i is the variable of frame
y_error = [] #initialize the err as 

x_line = [] #initialize the xdata 
y_line = [] #initialize the ydata  

def animate(i):
    global w, flag
    
    if flag == True and epoch <= max_epochs:
        flag = False

        for j in range(n):

            u = np.dot(x_b[j,:], w) #dot product of data with weights, the output function
            if u > 0: 
                y[j] = 1  

            else: 
                y[j] = 0

            if y[j] != d[j]: #check the error (difference between the output and the target)
                #print(f'\ndiff = {d[j] - y[j]}\n') #debug
                w += beta * (d[j] - y[j]) * x_b[j,:].reshape(3, 1) #calculate the new weights
                flag = True 

            y_error.append(int(d[j] - y[j])) 
            k.append(i)    
            
            #calculate the new cordinates of the dividing line
            x_line = np.array([np.min(x_b[:,1]), np.max(x_b[:,1])])
            y_line = (w[0] - w[1]*x_line)/w[2]
        #print(f'k = {k}, err = {y_error}') #debug
            
        line.set_data(x_line, y_line) 
        error.set_data(k, y_error)
        output_u.set_data(x_b[:,1], y)

    return line, output_u, error, 

'''
 You must store the created Animation in a variable 
that lives as long as the animation should run. 
Otherwise, the Animation object will be garbage-collected and the animation stops.
'''
anim1 = FuncAnimation(fig, animate, 
                    frames=max_epochs, 
                    interval=50,
                    blit=True)


plt.show()



