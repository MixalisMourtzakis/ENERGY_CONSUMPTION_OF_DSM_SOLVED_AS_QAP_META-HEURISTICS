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

#------------------------------------------FANT ALGORITHM-------------------------------------------------#


def FANT(n, A, Ta, E, nr_iterations):
   

    #--------------------------------------------FUNCTIONS NEEDED FOR THE FANT ALGORITHM-------------------------------#
    
    def unif(low, high):
        random_float = random.random()   # Generate A random float in the range [0, 1)
        return low + int((high - low + 1) * random_float)   # Scale it to the desired range and convert to an integer

    #CALCULATE EP FOR A GIVEN PERMUTATION p 
    def calculate_cost(n , p , A , Ta):    
            #CALCULATING PERMUTATION MATRIX BASED ON SPECIFIC PERMUTATION pe
            P = np.zeros((n, n))
            for j in range(n):
                P[j, p[j] - 1] = 1  # Permutation Matrix

            Ep = Vdd**2 * np.trace(A @ P @ Ta @ P.T)  # Evaluate Cost Function for this Permutation Matrix
            return Ep,p

    #LOCAL SEARCH AFTER THE RANDOM GENERATED SOLUTION
    def local_search(n, p , A , Ta):

        #Initialize Best_EP
        Best_Ep ,Best_p = calculate_cost(n , p , A , Ta)

        # Initialize the set of moves
        move = [0] * (n * (n - 1) // 2)
        nr_moves = 0

        # Populate the 'move' array
        for i in range(n - 1):
            for j in range(i + 1, n):
                move[nr_moves] = n * i + j
                nr_moves += 1

        for scan_nr in range(2):
        
            # Shuffle the moves
            for i in range(nr_moves - 1):
                j = unif(i + 1, nr_moves - 1)  
                move[i], move[j] = move[j], move[i] 

            for i in range(nr_moves):
                r = move[i] // n
                s = move[i] %n
                p[r], p[s] = p[s], p[r]
                Temp_Best_Ep,Temp_Best_p = calculate_cost(n , p , A , Ta)

                if Temp_Best_Ep < Best_Ep:
                    Best_Ep = Temp_Best_Ep
                    Best_p = Temp_Best_p
                    return Best_Ep , Best_p
        return Best_Ep , Best_p

       
    #CREATE SOLUTION BASED ON TRACE
    def Generate_solution_trace(n , p , trace):
        nexti = list(range(n))
        nextj = list(range(n))
        sum_trace = [sum(trace[i]) for i in range(n)]

        random.shuffle(nexti)
        random.shuffle(nextj)
        used_j=[]

        for i in range(n):
            target = unif(0, sum_trace[nexti[i]] - 1)
            j = i
            sum_val = trace[nexti[i]][nextj[j]]

            while sum_val < target and j in used_j:
                j += 1
                sum_val += trace[nexti[i]][nextj[j]]
                
            p[nexti[i]] = nextj[j]
            used_j.append(j)
            for k in range(i, n):
                sum_trace[nexti[k]] -= trace[nexti[k]][nextj[j]]
                nextj[j], nextj[i] = nextj[i], nextj[j]
        return p 

    #TRACE FUNCTIONS
    def init_trace(n, increment, trace):
        for i in range(n):
            for j in range(n):
                trace[i][j] = increment
        return trace

    def update_trace(n , p , best_p , increment , trace):
        i = 0
        R=1
        while i < n and p[i] == best_p[i]:
            i += 1

        if i == n:
            # All elements of p are equal to best_p
            increment += 1  
            init_trace(n, increment, trace)
        else:
            # Update trace based on differences between p and best_p
            for i in range(n):
                trace[i][p[i]] += increment[0]
                trace[i][best_p[i]] += R
        return trace

   #--------------------------------------INITIALIZATION OF VARIABLES---------------------------------------------#
    trace = np.zeros((n, n), dtype=int) #Initialize trace
    p = list(range(n))  # Initialize p
    Ep = E
    best_Ep = E  #Initialize best E
    best_p = p  #Initialize best p
    increment =1 #Initialize Increment
    init_trace(n, increment, trace) #Initialize Trace

    #Initialize PLOT
    Start_time=time.time()
    time_running=[0,0]
    Ep_plot = [Ep,Ep]
    Iteration_plot = [0,0]
    fig, ax1 = plt.subplots()
    #--------------------------------------------FANT ALGORITHM------------------------------------------------------#

    
    # FANT iterations
    for no_iteration in range(1, nr_iterations + 1):

        p = Generate_solution_trace(n, p, trace)    #Create solution based on trace

        Ep , p  = local_search(n, p , A , Ta)    #Trying to improve solution using local search

        # Check if a better Ep is found
        if Ep < best_Ep:

            best_Ep = Ep #Update best Ep

            #Update plot lists
            time_running.append(time.time()-Start_time)
            Ep_plot.append(best_Ep)
            Iteration_plot.append(no_iteration)

            print(f"New best solution value, Ep: {Ep} Found at iteration: {no_iteration}")

            best_p = p
            increment = 1
            init_trace(n, increment, trace)  #Re-initialize trace if a better Ep is found
        else:
            update_trace(n, p, best_p, increment, trace)  #Update trace if a better Ep is not found in this iteration

    print("The best Ep is", best_Ep,"found in",time.time()-Start_time,"seconds using the FANT Algorithm")

    #Update plot lists
    time_running.append(time.time()-Start_time)
    Ep_plot.append(best_Ep)
    Iteration_plot.append(nr_iterations)

    #Create plot for visualization of our results
    ax1.plot(time_running, Ep_plot,'w-')
    ax2 = ax1.twiny()
    ax2.plot(Iteration_plot, Ep_plot , '-o')

    plt.title('FANT ALGORITHM')
    ax1.set_xlabel('Time running(seconds)')
    ax1.set_ylabel('Ep value')
    ax2.set_xlabel('Number of iterations')

   
    plt.show()
    return time_running , Ep_plot

FANT(n,A,Ta,E,800000)