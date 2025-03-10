
# coding: utf-8

# This program numerically solves flavor evolution of neutrinos and anti-neutrinos emmitted from an infinite line sources. The system, that is, occupation numbers and fluxes are assumed to be stationary and homogeneous.
# The partial differential equation describing this setup is as follows 
# \begin{equation}
# i(v_x \partial_x + v_z \partial_z)\rho_{E,\textbf{v}} = [H, \rho_{E,\textbf{v}}]
# \end{equation}
# where 
# \begin{equation}
#     H = \frac{M^2}{2E} + v^\mu \Lambda_\mu + \int d\Gamma ' \, v ^\mu v'_\mu \rho_{E', \textbf{v'}}
# \end{equation}
# 
# An infinite source cannot be put into a computer so we impose a periodic boundary condition.

# In[187]:


import matplotlib
# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from scipy.integrate import ode
from scipy.integrate import complex_ode
import matplotlib.pyplot as plt
import matplotlib.lines as mline
from numpy import seterr, isneginf
from scipy import fftpack as sp
# import seaborn as sns


# Following part describes the $M^2$ matrix.
# 'theta' is the vacuum mixing angle.

# In[209]:


m1 = 0
m2 = 9*10**-5
theta = 0.001
U = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
vacuum = np.dot(U, np.dot(np.diag([m1,m2]), np.transpose(U)))
#vacuum = np.diag(np.diag(vacuum))
print(vacuum)


# The matter term assuming that vector part of $\Lambda$ is zero.

# In[210]:


factor = 0
n_e = 1
n_mu = 0.0
matter = factor*np.diag([n_e, n_mu])


# All the modes present in the system.
# 'mu' represents the density of the neutrinos.
# 'Kin_vars' stores energy and the velocity of all the modes.
# 'Vel_arr_4' has four velocity of all the modes.

# In[211]:


mu = 1
Nmodes = 40
v_z_arr = np.linspace(0, 1, 20)

Kin_vars = np.ones((Nmodes,3)) # eneergy, v_x, v_z
Kin_vars[:20, 1] = v_z_arr
Kin_vars[20:, 0] = -np.ones(20)
Kin_vars[20:, 1] = v_z_arr
Kin_vars[:, 2] = np.zeros(40)
#print(Kin_vars)
#Nmodes = 4
#Kin_vars = np.ones((Nmodes,3)) # eneergy, v_x, v_z
#Kin_vars[0] = [1, 0.5, 0]
#Kin_vars[1] = [-1, 0.5, 0]
#Kin_vars[2] = [1, 0.8, 0]
#Kin_vars[3] = [-1, 0.8, 0]
Vel_arr_4 = Kin_vars.astype(np.complex64) # i, v_x, v_z
Vel_arr_4[:,0] = 1j
print(Kin_vars)
# print(Vel_arr_4)


# The initial occupations numbers for all the modes. (Can be read from a data file)

# In[212]:


occupations = np.zeros((Nmodes,2))

for i in range(Nmodes):
	if (i<10):
		occupations[i] = [0.6, 0]
	elif(i<20):
		occupations[i] = [2.4, 0]
	else:
		occupations[i] = [-1, 0]
print(occupations)
#occupations[0] = [1, 0]
#occupations[1] = [-0.5, 0]
#occupations[2] = [0.5, 0]
#occupations[3] = [-1, 0]
init_dens = []
for i in range(Nmodes):
    init_dens.append(np.reshape(np.diag(occupations[i]), -1))
init_dens = np.array(init_dens).astype(np.complex64)
print(init_dens)


# Discretizing in $x$ and defining initial density matrix with seeded perturbations at $x_i's$.

# In[213]:


L = 200
Ndivs = 1000
deltax = L/Ndivs
x_arr = np.arange(0, L, deltax)

def aise(i):
    if (i<10):
        return 0.3/2
    elif (i<20):
        return 1.2/2
    else:
        return -1/2
    
def perturb(mat, div_num):
    if 1:#np.abs(div_num-500)<10:
        N_mats = np.reshape(mat, (Nmodes, 2, 2)).astype(np.complex64)
        for i in range(Nmodes):
    #         print(x_arr[div_num])
            N_mats[i][0][1] = aise(i)*(1/np.sqrt(2*3.14))*(10**-6)*np.exp(-0.5*(x_arr[div_num] - L*0.5)**2)*(1+1j) #*(np.random.uniform(-0.005,0.005) + 1j*np.random.uniform(-0.005,0.005))#/10**max(np.abs(div_num-500), 5)
            N_mats[i][1][0] = np.conj(N_mats[i][0][1])
        return np.reshape(N_mats, -1)
    return np.reshape(mat, -1)

tot_init_vec = []
for i in range(Ndivs):
    tot_init_vec.extend(perturb(init_dens, i))
tot_init_vec = np.array(tot_init_vec)
print(tot_init_vec[80002])
tot_init_vec = np.reshape(tot_init_vec, -1)


# Vacuum and matter hamiltonians at each of the $x_i's$ for all modes.

# In[214]:


vacPart = []
for t1 in range(Ndivs):
    for t2 in range(Nmodes):
        vacPart.append(vacuum/Kin_vars[t2, 0])
        
vacPart = np.reshape(np.array(vacPart), (Ndivs, Nmodes, 2, 2))
print(vacPart.shape)
# print(vacPart)

matterPart = np.zeros((Ndivs, Nmodes, 2, 2))
for t1 in range(Ndivs):
    for t2 in range(Nmodes):
        matterPart[t1,t2] = matter
    
#matterPart = np.reshape(np.array(matterPart), (Ndivs, Nmodes, 2, 2))
print(matterPart.shape)
# print(matterPart)


# After discretiziation in x the differential equations become
# \begin{equation}
#     v_z \partial_z \rho_{E, \textbf{v}}(x_i) = -i [H, \rho_{E, \textbf{v}}] - v_x \frac{ \rho_{E, \textbf{v}}(x_{i+1}) - \rho_{E, \textbf{v}}(x_{i-1})}{2 \Delta x}   
# \end{equation}
# The next piece is the rhs function of the above equation for all the modes and points.
# 'vel_tensor' contains $\int d\Gamma v_\mu \rho_{E, \textbf{v}}(x_i)$ at all the points on x axis, so it is a (Ndiv x 3 x 2 x 2) tensor.
# 'SelfPart' is $\int d\Gamma' \, v^\mu v'_\mu \rho_{E', \textbf{v'}}(x_i)$.
# 

# In[229]:


def ConvToMat(arr):
    return np.reshape(arr, (2,2))

def func(t, tot_vec):
    tot_mat = np.reshape(tot_vec, (Ndivs, Nmodes, 2, 2)) #reshaping the state vector into a tensor containing 2x2 density matrices for all the modes and at each point
    vel_tensor = np.einsum('ij,ki...->kj...', Vel_arr_4, tot_mat)
    SelfPart = -np.einsum('ij,kj...->ki...', Vel_arr_4, vel_tensor)
    Ham_array = mu*SelfPart+matterPart+vacPart
    Commutators = -1j*(np.einsum('ijkl,ijlm->ijkm', Ham_array, tot_mat) - np.einsum('ijkl,ijlm->ijkm', tot_mat, Ham_array))
#    Commutators1 = np.einsum('ij...,j->ij...', Commutators, 1/(Vel_arr_4[:,2]))
    gradient = np.zeros((Ndivs, Nmodes, 2, 2)).astype(np.complex64)
    for i in range(Nmodes):
        for j in range(2):
            for k in range(2):
                gradient[:, i, j, k] = sp.diff(tot_mat[:, i, j, k], period = L)
#    gradient[1:Ndivs-1] = (tot_mat[2:Ndivs] - tot_mat[1:Ndivs-1])/(1*deltax)
#    gradient[0] = (tot_mat[1] - tot_mat[0])/(1*deltax)
#    gradient[-1] = (tot_mat[0] - tot_mat[Ndivs-1])/(1*deltax)
    gradient1 = np.einsum('ij...,j->ij...', gradient, Vel_arr_4[:,1])
    
    answer = np.reshape(-gradient1 + Commutators, -1)
    return answer
#
#def jacobian(t, y):
    
# The main body of the code.
# All the data is stored in 'arr'.

# In[246]:
#print(np.imag(tot_init_vec))

solver = ode(func).set_integrator(name = 'zvode', method = 'BDF', nsteps = 1000000, atol = 0.000001)
solver.set_initial_value(tot_init_vec, 0)
z_final = 8
deltaz = 0.1
z_arr = np.arange(0, z_final, deltaz)

arr = []
for i in range(len(z_arr)):
    if(i%10 == 0):
        print(i)
    solver.integrate(solver.t+deltaz)
    arr.append(solver.y)#[:Ndivs*Nmodes*2*2] + 1j*solver.y[Ndivs*Nmodes*2*2:])
arr = np.array(arr)    

# To present the off-diagonal element of one of the modes as a heatmap.

# In[267]:

seterr(divide= 'ignore')
arr = np.reshape(arr, (len(z_arr), Ndivs, Nmodes, 2, 2))
print(arr.shape)
#offdiag_arr = np.abs(np.sum(arr[:, :, :, 0, 1], axis=2))

offdiag_arr = np.log10(np.abs(np.sum(arr[:, :, :, 0, 1], axis=2)))
offdiag_arr[isneginf(offdiag_arr)] = -100
print(offdiag_arr.shape)
plt.figure(1)
x_data, z_data = np.meshgrid(x_arr, z_arr)
plt.pcolormesh(x_data, z_data, offdiag_arr, cmap = plt.get_cmap("Spectral"))
plt.colorbar()
plt.clim(-7,0)
#cs = plt.contour(x_data, z_data, offdiag_arr, np.linspace(-8,-7.99,1), colors="k")
plt.xlabel("x")
plt.ylabel("t")
plt.grid(1)
plt.xticks(np.arange(0, L, L/10))

plt.figure(2)
plt.plot(z_arr, offdiag_arr[:,5])

