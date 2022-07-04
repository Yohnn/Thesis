#import necessary libraries and functions
import numpy as np
from matplotlib import pyplot as plt
from numba import vectorize,jit,cuda,guvectorize,float32, njit, prange
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats
import scipy.special as sc
import time

np.random.seed(42)
print('Start')
#define the paramters of the boundaries and create the general boundary
L = 1 # x-length of one sphere of the channel
a = 0.1*L #half width of the bottleneck width 
l = L/2 #half x-length 
R = np.sqrt(l**2 + a**2) #radius


x_coordinates = np.linspace(0,1,500).astype(np.float32) #define x values for only one sphere
x_bar = np.mod(x_coordinates,L) #modulo division equivalent of x to ensure periodicity of the channel

#create general boundary
y_u = [] 

for i in x_bar:
  y_u.append(np.sqrt(R**2 - (i - l)**2))

y_u = np.array(y_u).astype(np.float32)
y_l = -y_u

#define particle sizes; will be used for the simulation and creation of r-dependent effective boundaries
r_p1 = 0.1*a
r_p3 = 0.3*a
r_p5 = 0.5*a
r_p7 = 0.7*a
r_p9 = 0.9*a

#create the different effective boundaries per particle size

N = 1000 #number of particles to be used in the simulation
y_range = np.linspace(-R,R,N) #possible y values

#creating the different r-dependent effective boundaries
'''
y_u_eff1 = []
y_u_eff3 = []
y_u_eff5 = []
y_u_eff7 = []
y_u_eff9 = []

effective_boundaries = [y_u_eff1,y_u_eff3,y_u_eff5,y_u_eff7,y_u_eff9]
particle_sizes = np.array([r_p1,r_p3,r_p5,r_p7,r_p9])

for boundary,size in zip(effective_boundaries,particle_sizes):
  L_p = size*l/R
  U_p = L- L_p
  for i in x_bar:
    if i < L_p:
      boundary.append(-np.sqrt(size**2 - i**2) + a)
    elif i < U_p:
      boundary.append(np.sqrt((R - size)**2 - (i - l)**2))
    elif i < L:
      boundary.append(-np.sqrt(size**2 - (i - L)**2) + a)

y_u_eff1 = np.array(y_u_eff1).astype(np.float32)
y_u_eff3 = np.array(y_u_eff3).astype(np.float32)
y_u_eff5 = np.array(y_u_eff5).astype(np.float32)
y_u_eff7 = np.array(y_u_eff7).astype(np.float32)
y_u_eff9 = np.array(y_u_eff9).astype(np.float32)

y_l_eff1 = -y_u_eff1
y_l_eff3 = -y_u_eff3
y_l_eff5 = -y_u_eff5
y_l_eff7 = -y_u_eff7
y_l_eff9 = -y_u_eff9
'''

#plot the effective boundaries to observe if correct
'''
plt.figure(figsize=(5, 5), dpi=80)
plt.plot(x_coordinates,y_u,'k')
plt.plot(x_coordinates,y_l,'k')
plt.plot(x_coordinates,y_u_eff1,'b--', label = '$r_p = 0.1a$')
plt.plot(x_coordinates,y_l_eff1,'b--')

plt.plot(x_coordinates,y_u_eff3,'r--',label = '$r_p = 0.3a$')
plt.plot(x_coordinates,y_l_eff3,'r--')

plt.plot(x_coordinates,y_u_eff5,'y--',label = '$r_p = 0.5a$')
plt.plot(x_coordinates,y_l_eff5,'y--')

plt.plot(x_coordinates,y_u_eff7,'m--',label = '$r_p = 0.7a$')
plt.plot(x_coordinates,y_l_eff7,'m--')

plt.plot(x_coordinates,y_u_eff9,'g--',label = '$r_p = 0.9a$')
plt.plot(x_coordinates,y_l_eff9,'g--')

plt.legend(loc = 'best')
plt.show()
'''

#defining the variables that will hold the position of each particle
particles_x = np.zeros(N).astype(np.float32) + L/2 #x coordinate
particles_y = np.zeros(N).astype(np.float32)       #y coordinate
particles_xcum = np.zeros(N).astype(np.float32)   #cumulative x distance travelled

#arrays to hold the new position for each particle
new_particles_x = np.zeros(N).astype(np.float32)
new_particles_y = np.zeros(N).astype(np.float32)
new_particles_xcum = np.zeros(N).astype(np.float32)
t_counter = 0 #counts cumulative time

'''
particles_x_device = cuda.to_device(particles_x)
particles_y_device = cuda.to_device(particles_y)
particles_xcum_device = cuda.to_device(particles_xcum)

new_particles_x_device = cuda.device_array_like(particles_x)
new_particles_y_device = cuda.device_array_like(particles_x)
new_particles_xcum_device = cuda.device_array_like(particles_x)
'''

#defining the functions 
'''
This code uses Numba which compiles the python code to an efficient
machine code that will run much faster. In using Numba, the main simulation
will be wrapped by a function to utilize.
'''

#F(y,r) - There is a 'cap' argument which limits the magnitude of the force
#to maintain optimal simulation runtime.




@njit
def logforce(y,cap,r):
  if y == 0:
    return 0
  A = 1
  alpha = np.log(np.abs(y**2 - R**2))/2
  delta = b/(y**2)
  numerator = -a*y*(alpha + delta)
  denominator = r*(np.abs(y)**2)
  force = numerator/denominator
  if force > cap:
    return cap
  elif force < -cap:
    return -cap
  else:
    return force

@njit
def move_x(position):
  return position + fs*h + np.sqrt((2*a*h)/r_p)*np.random.normal(0,1)

@njit
def move_y(position,cap,r):
  return position + logforce(position,cap,r)*h + np.sqrt((2*a*h)/r_p)*np.random.normal(0,1)

@njit
def new_position(x,y,x_cum,cap,r):
  x_new = move_x(x)
  y_new = move_y(y,cap,r)
  x_inc = x_new - x

  #apply periodic boundary condition along the x direction
  if x_new > x_coordinates[-1]: #exits the right
    x_new = x_coordinates[0] + np.abs(x_new - x_coordinates[-1])
  elif x_new < x_coordinates[0]: #exits the left
    x_new = x_coordinates[-1] - np.abs(x_new - x_coordinates[0])

  x_index = np.abs(x_coordinates - x_new).argmin()

  reduce_cap_counter = 0 #counter to reduce the cap 
  reduce_cap_limit = 10 #if the while loop loops for the tenth time, the cap will be reduced
  while (y_new > y_u_eff[x_index]) or (y_new < y_l_eff[x_index]):
    x_new = move_x(x)
    y_new = move_y(y,cap,r)
    x_inc = x_new - x
    
    #apply periodic boundary condition along the x direction
    if x_new > x_coordinates[-1]: #exits the right
      x_new = x_coordinates[0] + np.abs(x_new - x_coordinates[-1])
    elif x_new < x_coordinates[0]: #exits the left
      x_new = x_coordinates[-1] - np.abs(x_new - x_coordinates[0])

    x_index = np.abs(x_coordinates - x_new).argmin()

    '''
    code to reduce cap if the simulation won't exit the while loop:
    Not exiting the while loop means that the force produced is too large and the new particle positions are always
    generated outside the boundary. Reducing the cap would allow the simulation to continue while 
    expressing the largest possible force that would allow the simulation to continue 
    '''
    '''if reduce_cap_counter == reduce_cap_limit: #once the while loop has loop for n times
      cap -= 100 #reduce cap by a magnitude of 100
      reduce_cap_counter = 0 #reset the counter

    reduce_cap_counter += 1 #increment counter'''

    
  x_cum += x_inc

  #accept new position
  return x_new,y_new,x_cum

@njit(parallel = True)
def simulation_timestep(particles_x,particles_y,particles_xcum,new_particles_x,new_particles_y,new_particles_xcum,cap,r):
  for i in prange(len(particles_x)):
      new_particles_x[i],new_particles_y[i],new_particles_xcum[i] = new_position(particles_x[i],particles_y[i],particles_xcum[i],cap,r)
      particles_x[i],particles_y[i],particles_xcum[i] = new_particles_x[i],new_particles_y[i],new_particles_xcum[i]

@njit
def main_simulation(particles_x,particles_y,particles_xcum,new_particles_x,new_particles_y,new_particles_xcum,mean_x_cum,mean_v_cum,t,cap,r):
  for step in range(int(N_steps)):
    simulation_timestep(particles_x,particles_y,particles_xcum,new_particles_x,new_particles_y,new_particles_xcum,cap,r)
    mean_x_cum[step] = np.mean(particles_xcum)
    


########################INPUT VARIABLES HERE#############################

######################
######################
    
cap = 40000#cap for the force to prevent reaching infinity

fs = 0 #static force
t = 1  ### simulation time
h = 1e-5 #time step
N_steps = t/h #total number of steps

#FOR r and r_p : [r_p1, r_p3, r_p5, r_p7, r_p9]
#FOR y_u_eff : [y_u_eff1, y_u_eff3, y_u_eff5, y_u_eff7, y_u_eff9]
#FOR y_l_eff : [y_l_eff1, y_l_eff3, y_l_eff5, y_l_eff7, y_l_eff9]
#FOR b : [10, 1, 0, 0.1, 0.01, 0.001, 0.0001, 0.00001]

b = 1 #gap width parameter ####### CHANGE THIS ########
r = r_p1 #particle size argument for the force ######### CHANGE THIS #########
r_p = r 
y_u_eff = y_u 
y_l_eff = y_l                    

time_steps = np.linspace(0,t,int(N_steps))

mean_x_cum = np.zeros(shape = (int(N_steps)), dtype = np.float32) #array for the mean cumulative x travelled
mean_v_cum = np.zeros(shape = (int(N_steps)), dtype = np.float32) #array for the mean velocities

#######################
#######################


################# MAIN SIMULATION ######################

start = time.time() #mark start of simulation

main_simulation(particles_x, particles_y, particles_xcum, new_particles_x, new_particles_y, new_particles_xcum, mean_x_cum,mean_v_cum,t,cap,r)

end = time.time() #end of simulation

print("Simulation Time (in minutes): ", (end-start)/60)
np.savetxt("probdist_b0000001_rp5.csv",particles_y,delimiter=",") ################ CHANGE THIS ###############

################# END SIMULATION ########################

################# START OF STATISTICAL ANALYSIS #################

#function to solve for the KS-test statistic D

def prob_dist(y,b):
  parenth_left = R*np.exp(-b/(R**2))
  parenth_right = np.sqrt(np.pi)*np.sqrt(b)*sc.erfc(np.sqrt(b)/R)
  factor = 2*np.exp(b)*(parenth_left - parenth_right)
  A_norm = 1/factor
  inside_term = b*(y**2 - 1)/(y**2)
  return A_norm*np.exp(inside_term)

def create_cdf(y,b):
    pdf = prob_dist(y,b)
    cdf = np.cumsum(pdf)

    #then normalize
    cdf = cdf/cdf[-1]
    return cdf

def ks_test(emp,parent):
  difference = np.abs(emp-parent)
  return np.max(difference)

#define parameters
alpha = 0.05
p_value = 1.36/np.sqrt(N) #based on the KS-test table of p values

#first, create the theoretical pdf and cdf given b and r
cdf_theo = create_cdf(y_range,b)

#then, create empirical cumulative distribution function of the data
emp_data = particles_y #store the particle positions in an array
ecdf = ECDF(particles_y) #callable, returns percentage of values below the y (ECDF)
ecdf_data = ecdf(y_range) #actual emprical cdf

#plot and observe distribution
plt.figure(figsize=(10, 10), dpi=80)
plt.plot(y_range,cdf_theo,'r',label = 'theoretical cdf $C(y)$')
plt.plot(y_range,ecdf_data,'b', label = 'empirical cdf $G_n(y)')
plt.title('$b = 10$, $r_p = 0.1a$')
plt.show()

#compute for test statistic and conclude if accept or reject H0
#H0: F_exp = F_theo (the experimental data obeys the theoretical distribution)

D = ks_test(ecdf_data, cdf_theo)
print('KS-test statistic D: ', D, " p_value = ", p_value)

if D < p_value:
    print("D < p_value. Therefore, ACCEPT Ho")
else:
    print("D > p_value. Therefore, REJECT Ho")
#########################################################

#data processing

#mean_v_cum = (mean_x_cum[1:] - 0.5)/time_steps[1:]

plt.figure(figsize=(8, 8), dpi=80)
plt.plot(x_coordinates,y_u,'b')
plt.plot(x_coordinates,y_l,'b')

plt.plot(x_coordinates,y_u_eff,'r--')
plt.plot(x_coordinates,y_l_eff,'r--')

plt.scatter(new_particles_x,new_particles_y, s = 2,c = 'red')
#plt.xlim(0,1)
plt.show()

'''
plt.plot(time_steps,mean_x_cum)
plt.plot(time_steps[1:],mean_v_cum)
#plt.xlim(0.1,1)
plt.ylim(-3,3)
plt.show()
'''

print("Gap width parameter b = ",b)
'''
print("Mean velocity <v> = ",np.mean(mean_v_cum[-1000:]))
print("Mean velocity <v> (last) = ", mean_v_cum[-1])
'''

