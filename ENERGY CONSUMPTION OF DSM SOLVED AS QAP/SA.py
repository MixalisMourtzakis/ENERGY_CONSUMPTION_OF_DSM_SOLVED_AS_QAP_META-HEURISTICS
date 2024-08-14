import numpy as np
import math
from itertools import permutations
import time
import random
import matplotlib.pyplot as plt



#INITIALIZE THE PARAMETERS USED
Vdd = 1
CL = 1
Lambda = 3
m = 10000   #(nxm) is the size of the data matrix (n = number of lines),(m = number of periods)
n=50

#-------------------------------------------------CALCULATING (A,R0,R1,TA,E)---------------------------------------------------------#

#Calculate A matrix
h0 = np.eye(n)   # Create h0 matrix (identity matrix)

h1 = np.diag(np.ones(n-1), k=1) + np.diag(np.ones(n-1), k=-1)  # Create h1 
h1 += np.diag(np.zeros(n))

A = ((1 + 2 * Lambda) * h0 - Lambda * h1) * CL   #Create A matrix

#Create random L matrix(MATRIX OF OUR DATA-RANDOM MATRIX EVERY TIME SO THAT OUR RESULTS ARE NOT DEPENDED ON A SPECIFIC DATASET) 
L = np.random.randint(0, 2, size=(n, m))

# Calculate R0 matrix
h = np.zeros((n, n))
for k in range(m):
    h += np.outer(L[:, k], L[:, k])
R0 = h / m

# Calculate R1 matrix
h = np.zeros((n, n))
for k in range(m - 1):
    h += np.outer(L[:, k + 1], L[:, k])
R1 = h / (m - 1)

# Calculate Transition Activity Matrix (Ta)
Ta = R0 - (R1 + R1.T) / 2

# Calculate Expected Energy per Transition (E)
E = Vdd**2 * np.trace(A @ Ta)


#------------------------------------------------SIMULATED ALGORITHM---------------------------------------------------------------------#
def Simulated_Annealing(n, A ,Ta , E, initial_delta, nr_iterations):

    #-----------------------------------------------------------------Functions-------------------------------------------------------#

    #Randomly swap 2 elements
    def swap_random(seq):
        idx = range(len(seq))
        i1, i2 = random.sample(idx, 2)
        seq[i1], seq[i2] = seq[i2], seq[i1]

    #CALCULATE EP FOR A GIVEN PERMUTATION p 
    def calculate_cost(n , p , A , Ta):    
            #CALCULATING PERMUTATION MATRIX BASED ON SPECIFIC PERMUTATION pe
            P = np.zeros((n, n))
            for j in range(n):
                P[j, p[j] - 1] = 1  # Permutation Matrix

            Ep = Vdd**2 * np.trace(A @ P @ Ta @ P.T)  # Evaluate Cost Function for this Permutation Matrix
            return Ep,p
    

    #------------------------------------------------------------------------INITIALIZATION------------------------------------------------------#
    
    Ep_best = E  #Initialize Ep_best 
    p = list(range(n))  # Initialize p 
    p_best , p_to_check = p , p

    
    #Initialize PLOT
    Start_time = time.time()
    time_running = [0,0]
    Ep_plot = [E,E]
    Iteration_plot = [0,0]
    fig, ax1 = plt.subplots()

    #-------------------------------------------------------------SA MAIN-------------------------------------------------------------#
    for i in range(nr_iterations):

        p_to_check == swap_random(p_to_check) #Randomly swap two elements of or current permutation p

        Temp_Best_Ep,Temp_Best_p = calculate_cost(n , p_to_check , A , Ta) #Calculating the cost for our current permutation p

        #Checking if we found better solution
        if Ep_best > Temp_Best_Ep:
            #Updating Ep_best-p_best
            p_best , p_to_check = Temp_Best_p , Temp_Best_p
            Ep_best = Temp_Best_Ep
            
            #Updating plot lists
            time_running.append(time.time()-Start_time)
            Ep_plot.append(Ep_best)
            Iteration_plot.append(i)
            print(f"New best solution value, Ep: {Ep_best} Found at iteration: {i}")
        else:
            
            delta = -math.log(initial_delta) / nr_iterations  #Calculating delta
            Temperature =0.1/(i + 1) #Calculating Temperature
            propability = (math.e)**(-delta/Temperature) #Calculating propability
            
            #Keeping the temp_Best_p with a propability of (propability)
            if random.random() <=propability:
                p_to_check = Temp_Best_p  #Edw eixa p !!!!!!!!!!!!!!!!!


    print("The best Ep is", Ep_best,"found in",time.time()-Start_time,"seconds using the SA Algorithm")

    #Updating plot lists
    time_running.append(time.time()-Start_time)
    Ep_plot.append(Ep_best)
    Iteration_plot.append(nr_iterations)

    #Creating plot for visualization of our results
    ax1.plot(time_running, Ep_plot,'w-')
    ax2 = ax1.twiny()
    ax2.plot(Iteration_plot, Ep_plot , '-o')

    plt.title(f'SA ALGORITHM (n={n})')
    ax1.set_xlabel('Time running(seconds)')
    ax1.set_ylabel('Ep value')
    ax2.set_xlabel('Number of iterations')

    
    plt.show()



Simulated_Annealing(n, A ,Ta , E, 0.00001, 100000000)    ##(0.00001, 1000000)