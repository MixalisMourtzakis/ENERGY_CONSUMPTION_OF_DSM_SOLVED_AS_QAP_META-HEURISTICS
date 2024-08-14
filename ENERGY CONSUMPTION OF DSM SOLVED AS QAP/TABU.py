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
n= 10

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


#----------------------------------------------------------------------------------------------------------------------------#

def TABU(n, A , Ta, E, nr_iterations , nr_of_trials , Aspiration ,Tabu_duration):

    #------------------------------------------------FUNCTIONS NEEDED----------------------------------------------------------#
    
    def Cube(x):    #Returns the cube of an number
        return x**3


    def generate_random_solution(n):       #Generate random solution using the random shufle of p
        p = list(range(n))
        random.shuffle(p)
        return p


    #CALCULATE EP FOR A GIVEN PERMUTATION p 
    def calculate_cost(n , p , A , Ta):    
            #CALCULATING PERMUTATION MATRIX BASED ON SPECIFIC PERMUTATION pe
            P = np.zeros((n, n))
            for j in range(n):
                P[j, p[j] - 1] = 1  # Permutation Matrix

            Ep = Vdd**2 * np.trace(A @ P @ Ta @ P.T)  # Evaluate Cost Function for this Permutation Matrix
            return Ep,p


    def compute_delta(n, A, Ta, p, i, j):
        d = (A[i][i] - A[j][j]) * (Ta[p[j]][p[j]] - Ta[p[i]][p[i]]) + \
            (A[i][j] - A[j][i]) * (Ta[p[j]][p[i]] - Ta[p[i]][p[j]])
        for k in range(n):
            if k != i and k != j:
                d += (A[k][i] - A[k][j]) * (Ta[p[k]][p[j]] - Ta[p[k]][p[i]]) + \
                    (A[i][k] - A[j][k]) * (Ta[p[j]][p[k]] - Ta[p[i]][p[k]])
        return d


    def compute_delta_part(A, Ta, p, delta, i, j, r, s):
        return delta[i][j] + (A[r][i] - A[r][j] + A[s][j] - A[s][i]) * \
            (Ta[p[s]][p[i]] - Ta[p[s]][p[j]] + Ta[p[r]][p[j]] - Ta[p[r]][p[i]]) + \
            (A[i][r] - A[j][r] + A[j][s] - A[i][s]) * \
            (Ta[p[i]][p[s]] - Ta[p[j]][p[s]] + Ta[p[j]][p[r]] - Ta[p[i]][p[r]])


   #----------------------------------------INITIALIZATION---------------------------------------------------------------------------------#
    Best_Ep = E #Initialize Ep_best 
    Best_p = list(range(n))  # Initialize Best_p

    #Initialize PLOT
    Start_time=time.time()
    time_running=[0,0]
    Ep_plot = [E,E]
    Resolutions_plot = [0,0]
    fig, ax1 = plt.subplots()


    for nr_resolutions in range(nr_of_trials):

        p = generate_random_solution(n)

        Temp_Ep , Temp_p = calculate_cost(n, p , A, Ta)

        delta = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                delta[i][j] = compute_delta(n,A, Ta, Temp_p,  i, j)

        #INITIALIZE TABU LIST
        Tabu_list = [[-(n * i + j) for j in range(n)] for i in range(n)]
              
        if Temp_Ep < Best_Ep:
            for current_iteration in range(1, nr_iterations + 1):
                i_retained , j_retained , min_delta = 999999999, 999999999,999999999
                already_aspired = False
                
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        authorized = (Tabu_list[i][Temp_p[j]] < current_iteration) or (Tabu_list[j][Temp_p[i]] < current_iteration)
                        aspired = (Tabu_list[i][Temp_p[j]] < current_iteration - Aspiration) or (Tabu_list[j][Temp_p[i]] < current_iteration - Aspiration) or (Temp_Ep < Best_Ep)
                        
                        if (aspired and not already_aspired and delta[i][j] < min_delta) or (not aspired and not already_aspired  and authorized and delta[i][j] < min_delta):
                            i_retained, j_retained = i, j
                            min_delta = delta[i][j]
                            if aspired:
                                already_aspired = True
                
                if i_retained == 999999999:
                    print("All moves are taTau!")
                else:
                    # Transpose elements at positions i_retained and j_retained
                    Second_Temp_p = Temp_p
                    Second_Temp_p[i_retained], Second_Temp_p[j_retained] = Temp_p[j_retained], Temp_p[i_retained]
                    Second_Temp_Ep , Second_Temp_p = calculate_cost(n, Second_Temp_p, A, Ta)

                    if Second_Temp_Ep < Temp_Ep:
                        Temp_Ep = Second_Temp_Ep
                        Temp_p = Second_Temp_p

                    # Update tabu list
                    Tabu_list[i_retained][Temp_p[j_retained]] = current_iteration + int(Cube(random.random()) * Tabu_duration)
                    Tabu_list[j_retained][Temp_p[i_retained]] = current_iteration + int(Cube(random.random()) * Tabu_duration)
                    
                    # Check if best solution improved
                    if Temp_Ep < Best_Ep:
                        Best_Ep = Temp_Ep
                        Best_p = Temp_p
                        
                        time_running.append(time.time()-Start_time)
                        Ep_plot.append(Best_Ep)
                        Resolutions_plot.append(nr_resolutions)
                        print(f"Solution of value: {Temp_Ep} found at iteration: {current_iteration} and resolution: {nr_resolutions}")

                    for i in range(n-1):
                        for j in range(i+1, n):
                            if i != i_retained and i != j_retained and j != i_retained and j != j_retained:
                                delta[i][j] = compute_delta_part(A, Ta, Temp_p, delta, i, j, i_retained, j_retained)
                            else:
                                delta[i][j] = compute_delta(n, A, Ta, Temp_p, i, j)


    print("The best Ep is", Best_Ep,"found in",time.time()-Start_time,"seconds using the TABU Algorithm")

    #Updating plot lists
    time_running.append(time.time()-Start_time)
    Ep_plot.append(Best_Ep)
    Resolutions_plot.append(nr_resolutions)
    
    #Creating plot for visualization of our results
    ax1.plot(time_running, Ep_plot,'w-')
    ax2 = ax1.twiny()
    ax2.plot(Resolutions_plot, Ep_plot , '-o')

    plt.title('TABU ALGORITHM')
    ax1.set_xlabel('Time running(seconds)')
    ax1.set_ylabel('Ep value')
    ax2.set_xlabel('Number of resolutions')

    # Display the plot
    plt.show()

TABU(n , A , Ta , E , 15000 , 80000 , 5*n *n , 8*n)