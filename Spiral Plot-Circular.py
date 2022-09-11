#!/usr/bin/env python
# coding: utf-8

# # Analytical

# In[1]:


n   = 200 # file number,
nr  = 638  # the radial grid in circular.par
ntheta = 1024 # the azimuthal grid number in circular.par

# ln(rmax / rmin) / 2pi * ntheta = nr


# In[ ]:


import math
import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir("/Users/RaymondMZhang/Downloads/fargo/outcircular")

# definitions
rmin = 0.2
r_p = 1
rmax = 3
dr = (rmax-rmin)/n
omega_p = 1
cs_p = 0.1
alpha = 1.5
beta = 0
theta_naught = 0
degree = math.pi/100

# reading planet data
planet = open('planet0.dat', 'r')

# finding planet point at 0
for line in planet: # Loop through every line in the file
    line = line.strip() # The strip() removes characters from both left and right 
    arr = line.split() # Split the string into many columns]
    arr = np.array(arr)
    arr = arr.astype(float)
    if arr[0] == n:
        planetx = arr[1]
        planety = arr[2]

# the radius of this circular orbit is just planetx since planety = 0
# radius = (planetx**2 + planety**2)**(0.5)
radius = 1

# arrays with the inner and outer points separately
# first n is which point the planet is positioned
# second n is how many points the spiral creates up to
# the 2 means each point has an (x,y)
innerpointarray = np.empty((n+1,n,2),float)
outerpointarray = np.empty((n+1,n,2),float)

# theta_naught is the angle of the planet position respect to the original point of the planet
for theta_naught in np.arange (n*degree,(n+1)*degree,degree):
    
    # planet changes position, so each new position is at (planetx, planety)
    planetx = radius * math.cos(theta_naught)
    planety = radius * math.sin(theta_naught)
    
    # plots the planet position with a black dot
    plt.plot(planetx,planety, color = 'black', marker = 'o')
    
    # x_inner and y_inner are the x and y points of the inner spiral
    # x_outer and y_outer are the x and y points of the outer spiral
    # x_inner, y_inner, x_outer, y_outer are not required for the overall code
#     x_inner = []
#     y_inner = []
#     x_outer = []
#     y_outer = []
    
    #drawing the spirals
    for t in np.arange (0,n*degree,degree):
        
        # the radius is changing at a rate of cs_p (- for inner and + for outer)
        r_inner = r_p - cs_p * t
        r_outer = r_p + cs_p * t
        print("r inner:", r_inner)
        print("r outer:", r_outer)
        
        # defining variables for clarity
        C = r_p * omega_p / cs_p
        a_inner = r_inner / r_p
        a_outer = r_outer / r_p
        f = beta - alpha + 1
        g = beta + 1
        
        # from previously solved equation (error for n > 10, I don't know why: Says theta_inner becomes infinity)
        theta_inner = theta_naught - C * ((a_inner**f/f - a_inner**g/g) - (1/f - 1/g))
        print("theta inner:", theta_inner)
        theta_outer = theta_naught + C * ((a_outer**f/f - a_outer**g/g) - (1/f - 1/g))
        
        # I have the values of the x and y coordinates put into multiple arrays for clarity
        # x_inner, y_inner, x_outer, y_outer are not required for the overall code
#         x_inner.append(r_inner * math.cos(theta_inner))
#         y_inner.append(r_inner * math.sin(theta_inner))
#         x_outer.append(r_outer * math.cos(theta_outer))
#         y_outer.append(r_outer * math.sin(theta_outer))
        
        print("theta:", theta_naught/degree)
        print("time: ", t/degree)
        innerpointarray[int(theta_naught/degree)][int(t/degree)][0] = (r_inner * math.cos(theta_inner))
        innerpointarray[int(theta_naught/degree)][int(t/degree)][1] = (r_inner * math.sin(theta_inner))
        outerpointarray[int(theta_naught/degree)][int(t/degree)][0] = (r_outer * math.cos(theta_outer))
        outerpointarray[int(theta_naught/degree)][int(t/degree)][1] = (r_outer * math.sin(theta_outer))
        

# graphing the points specifically
# i is which n (or planet position)
# j is which point the spiral at i creates
# i + j has to equal n-1
# for example, in this case, n = 10
# i starts as 0, requiring j to be 9
# repeats all the way until i is 9 and j is 0
#for i in range(n):
plt.contourf(xcoord, ycoord, rho, levels = levels)

i = n
for j in range(n):
    #if i + j == n-1:

    # plot the inner points as blue and the outer points as red
    # the point created for n = 10 in this case is just the black dot positioned where the planet is
    plt.scatter(innerpointarray[i][j][0],innerpointarray[i][j][1], color = 'blue', marker = 'o')
    plt.scatter(outerpointarray[i][j][0],outerpointarray[i][j][1], color = 'red', marker = 'o')

    

# rest of the plotting details
plt.xlim([-1.5,1.5])
plt.ylim([-1.5,1.5])
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.6, top=0.8)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()                 


# # Numerical

# In[ ]:


# read in data
import os
import math
get_ipython().run_line_magic('pylab', 'inline')


n2   = 200 # file number,
nr  = 638  # the radial grid in circular.par
ntheta = 1024 # the azimuthal grid number in circular.par

os.chdir("/Users/RaymondMZhang/Downloads/fargo/out2circular")
rho = fromfile("gasdens{0:d}.dat".format(n2), dtype='float64').reshape(nr, ntheta) #change dtype to 'float32' if your simulation is single precision
rho = np.transpose(rho)

file = open('used_rad.dat', 'r')
planet = open('planet0.dat', 'r')
rad = []

# ln(rmax / rmin) / 2pi * ntheta = nr
for line in file: # Loop through every line in the file
    line = line.strip() # The strip() removes characters from both left and right 
    columns = line.split() # Split the string into many columns
    rad = np.append(rad, float(columns[0])) # convert the column into floating numbers
    

for line in planet: # Loop through every line in the file
    line = line.strip() # The strip() removes characters from both left and right 
    arr = line.split() # Split the string into many columns]
    arr = np.array(arr)
    arr = arr.astype(float)
    if arr[0] == n2:
        planetx = arr[1]
        planety = arr[2]
#print(planetx, planety)
    

rad = (rad[1:] + rad[0:-1]) / 2. # change cell edge to cell center
theta = np.arange(0.0,2. * np.pi,2. * np.pi / ntheta)
radarr, thetaarr = np.meshgrid(rad,theta)


import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

zmin = 0.995
zmax = 1.005
levels = np.linspace(zmin, zmax, 200)
xcoord = radarr * np.cos(thetaarr)
ycoord = radarr * np.sin(thetaarr)
plt.contourf(xcoord, ycoord, rho, levels = levels)
# plt.contourf(rad, theta, rho, levels = levels)
#plt.plot(x,y)
# plt.plot(planetx,planety, color = 'black', marker = 'o')
#plt.xlim([-1.,0.5])
#plt.xlabel('log(r)')
#plt.ylim([0.,2*math.pi])
#plt.ylabel('theta')
plt.xlim([-2,2])
plt.xlabel('x')
plt.ylim([-2,2])
plt.ylabel('y')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import math
import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir("/Users/RaymondMZhang/Downloads/fargo/outcircular")

rmin = 0.2
r_p = 1
rmax = 3
dr = (rmax-rmin)/n
print(dr)

omega_p = 1
cs_p = 0.1
alpha = 1.5
beta = 0
theta_naught = 0

for deta in np.arange (0,math.pi/9,math.pi/9):
    planetx = radius * math.cos(deta)
    planety = radius * math.sin(deta)
    plt.plot(planetx,planety, color = 'black', marker = 'o')
    
    #drawing the spirals
    
    x = []
    y = []
    for r in np.arange (rmin, rmax, 0.1):
        if r > r_p:
            C = r_p * omega_p / cs_p
            a = r / r_p
            f = beta - alpha + 1
            g = beta + 1
            theta = theta_naught + C * ((a**f/f - a**g/g) - (1/f - 1/g)) + deta
            x.append(r * math.cos(theta))
            y.append(r * math.sin(theta))
        else:
            C = r_p * omega_p / cs_p
            a = r / r_p
            f = beta - alpha + 1
            g = beta + 1
            theta = theta_naught - C * ((a**f/f - a**g/g) - (1/f - 1/g)) + deta
            x.append(r * math.cos(theta))
            y.append(r * math.sin(theta))
    plt.plot(x,y)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.6, top=0.8)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()                 


# In[ ]:


import math
import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir("/Users/RaymondMZhang/Downloads/fargo/outcircular")

rmin = 0.2
r_p = 1
rmax = 3
dr = (rmax-rmin)/n
print(dr)

omega_p = 1
cs_p = 0.1
alpha = 1.5
beta = 0
theta_naught = 0

# x = np.empty((n),int)
# x = []
# y = np.empty((n),int)
# y = []

timearray = np.empty((n,n),int)
for i in np.arange(n):  # the n that incited the spiral
    for j in np.arange(n):  # the step
        timearray[i][j] = i+j
planet = open('planet0.dat', 'r')

for line in planet: # loop through every line in the file
    line = line.strip() # the strip() removes characters from both left and right 
    arr = line.split() # split the string into many columns]
    arr = np.array(arr)
    arr = arr.astype(float)
    if arr[0] == n:
        planetx = arr[1]
        planety = arr[2]


innerpointarray = np.empty((2*n,n,2),float)
outerpointarray = np.empty((2*n,n,2),float)

radius = planetx
for deta in np.arange (0,2*(n)*math.pi/9,math.pi/9):
    planetx = radius * math.cos(deta)
    planety = radius * math.sin(deta)
    plt.plot(planetx,planety, color = 'black', marker = 'o')
    
    #drawing the spirals
    x_inner = []
    y_inner = []
    x_outer = []
    y_outer = []
    for t in np.arange (0,n,1):
        r_inner = r_p - cs_p * t
        r_outer = r_p + cs_p * t
        C = r_p * omega_p / cs_p
        a_inner = r_inner / r_p
        a_outer = r_outer / r_p
        f = beta - alpha + 1
        g = beta + 1
        theta_inner = theta_naught - C * ((a_inner**f/f - a_inner**g/g) - (1/f - 1/g)) + deta
        theta_outer = theta_naught + C * ((a_outer**f/f - a_outer**g/g) - (1/f - 1/g)) + deta
        x_inner.append(r_inner * math.cos(theta_inner))
        y_inner.append(r_inner * math.sin(theta_inner))
        x_outer.append(r_outer * math.cos(theta_outer))
        y_outer.append(r_outer * math.sin(theta_outer))
        innerpointarray[int(deta*9/math.pi)][t][0] = (r_inner * math.cos(theta_inner))
        innerpointarray[int(deta*9/math.pi)][t][1] = (r_inner * math.sin(theta_inner))
        outerpointarray[int(deta*9/math.pi)][t][0] = (r_outer * math.cos(theta_outer))
        outerpointarray[int(deta*9/math.pi)][t][1] = (r_outer * math.sin(theta_outer))
    
#     for i in range(10):
#         print(x_inner[i], y_inner[i])
#     print("")
        
#    plt.plot(x_inner,y_inner)
#    plt.plot(x_outer,y_outer)
    
print(innerpointarray)


i = 5
for j in range(i):
    print(j,i-j)
    plt.scatter(innerpointarray[i][i-j][0],innerpointarray[i][i-j][1])
    plt.scatter(outerpointarray[i][i-j][0],outerpointarray[i][i-j][1])
        

#print(len(x), len(y))
#plt.xlim(-0.01,0.01)
#plt.ylim(-0.01,0.01)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.6, top=0.8)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()                 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




