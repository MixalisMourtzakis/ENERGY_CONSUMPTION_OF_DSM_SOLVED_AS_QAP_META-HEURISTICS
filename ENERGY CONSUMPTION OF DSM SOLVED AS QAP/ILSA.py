import numpy as np
from itertools import permutations
import time
import random
import matplotlib.pyplot as plt



#INITIALIZE THE PARAMETERS USED
Vdd = 1
CL = 1
Lambda = 3
m=10000   #(nxm) is the size of the data matrix (n = number of lines),(m = number of periods)
n=20

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






#---------------------------------------ILSA ALGORITHM-----------------------------------------------------------------------#

def ILSA(n , A, Ta , E , nr_iterations , SHUFFLE_TOLERANCE , propability_of_change , Nr_Swaps_LI):

    #-----------------------------------------------------------------Functions-------------------------------------------------------#

    #CALCULATE EP FOR A GIVEN PERMUTATION p 
    def calculate_cost(n , p , A , Ta):     
            #CALCULATING PERMUTATION MATRIX BASED ON SPECIFIC PERMUTATION pe
            P = np.zeros((n, n))
            for j in range(n):
                P[j, p[j] - 1] = 1  # Permutation Matrix

            temp_Ep = Vdd**2 * np.trace(A @ P @ Ta @ P.T)  # Evaluate Cost Function for this Permutation Matrix
            return temp_Ep,p

    #LOCAL IMPROVEMENT FOR THE CURRENT p(Makes a number(Nr_Swaps_LI) of random swaps on the current p with a propability of(Propability of change))
    def local_improvement(p):
        if random.random() > propability_of_change:
            for i in range(Nr_Swaps_LI):
                len_of_p = range(len(p))
                a, b = random.sample(len_of_p, 2)
                p[a], p[b] = p[b], p[a]
        return p


    #------------------------------------------------------------------------INITIALIZATION------------------------------------------------------#

    p = list(range(n))  # Initialize p
    Best_Ep = E  #Initialize best temp_Ep
    count = 0 #Initialize count


    #Initialize plot lists
    Start_time = time.time()
    time_running=[0,0]
    Ep_plot = [E,E]
    Iteration_plot = [0,0]
    fig, ax1 = plt.subplots()

    #------------------------------------------------------------ILSA MAIN-------------------------------------------------------------#

    for i in range(nr_iterations):

        temp_p = local_improvement(p)  #Make a local improvement on the current p

        temp_Ep , temp_p = calculate_cost(n , temp_p ,A ,Ta) #Calculate the temp_Ep using the current p

        #Check if better solution is found
        if  temp_Ep < Best_Ep:
            #Update best values
            Best_Ep = temp_Ep
            Best_p = temp_p

            #Update plot lists
            time_running.append(time.time()-Start_time)
            Ep_plot.append(Best_Ep)
            Iteration_plot.append(i)

            print(f"New best solution value, temp_Ep: {temp_Ep} Found at iteration: {i}")
            count = 0
   
        else:
            #If a better solution is not found for a number of iterations(SHUFFLE_TOLERANCE) then randomly shuffle our current p
            count += 1
            if count > SHUFFLE_TOLERANCE:
                random.shuffle(temp_p)
                count = 0

    #Update plot lists 
    time_running.append(time.time()-Start_time)
    Ep_plot.append(Best_Ep)
    Iteration_plot.append(nr_iterations)
    print("The best temp_Ep is", Best_Ep,"found in",time.time()-Start_time,"seconds using the ILSA Algorithm")

    #Create plot to visualize our results
    ax1.plot(time_running, Ep_plot,'w-')
    ax2 = ax1.twiny()
    ax2.plot(Iteration_plot, Ep_plot , '-o')

    plt.title('ILSA ALGORITHM')
    ax1.set_xlabel('Time running(seconds)')
    ax1.set_ylabel('temp_Ep value')
    ax2.set_xlabel('Number of iterations')

    plt.show()
        
        


ILSA(n,  A , Ta , E , 1000000 , 100 ,0.5 , 3)




######GENERAL COMMENTS#############
#The ilsa was executed for this values(SHUFFLE_TOLERANCE=100 , propability_of_change=0.5 , Nr_Swaps_LI=3) and the results where very prommising 
#when being compared with other heuristic algorithms 
