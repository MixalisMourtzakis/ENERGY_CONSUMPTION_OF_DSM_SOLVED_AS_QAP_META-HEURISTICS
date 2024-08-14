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
m = 30721  #(nxm) is the size of the data matrix (n = number of lines),(m = number of periods)  #BE CAREFULL IN CASE OF A SPECIFIC DATASET 
n = 24                                                                                 #MAKE SURE THAT THE n, m ARE RIGHT

#-------------------------------------------------CALCULATING (A,R0,R1,TA,E)---------------------------------------------------------#

#Function to read a specific text file in order to get a data set
def read_binary_matrix_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        matrix = []
        for line in file:
            # Remove whitespace and split each character into an integer
            row = [int(char) for char in line.strip()]
            matrix.append(row)
    return matrix



file_path = r"C:\Users\Μιχάλης Μουρτζάκης\Desktop\Διπλωματική\data_Input\Sine_24_bit.txt"
matrix = read_binary_matrix_from_file(file_path)



#Calculate A matrix
h0 = np.eye(n)   # Create h0 matrix (identity matrix)

h1 = np.diag(np.ones(n-1), k=1) + np.diag(np.ones(n-1), k=-1)  # Create h1 
h1 += np.diag(np.zeros(n))

A = ((1 + 2 * Lambda) * h0 - Lambda * h1) * CL   #Create A matrix

#Create random L matrix(MATRIX OF OUR DATA-RANDOM MATRIX EVERY TIME SO THAT OUR RESULTS ARE NOT DEPENDED ON A SPECIFIC DATASET) 
#L = np.random.randint(0, 2, size=(n, m))  #IF U WANT A RANDOM DATA SET UNCOMMENT THIS LINE AND PUT ABOVE THE SPECIFIC n,m YOU WANT

L = np.array(matrix)    
L = L.T

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



#---------------------------------------------------EXHAUSTIVE SEARCH ALGORITHM-------------------------------------------------------#
def exhaustive_search(n , A , Ta , E):
    Start_time=time.time()   #INITIALIZE START TIME

    
    pe = np.array(list(permutations(range(1, n+1)))) #CREATING A LIST OF ALL POSSIBLE PERMTATION OF n

    Ep_best = E  #Initialize the best Ep value
    p_best = list(range(n)) #Initialize permutation p that gives the best Ep

    for i in range(len(pe)):

        #CALCULATING PERMUTATION MATRIX BASED ON SPECIFIC PERMUTATION pe
        p = pe[i, :]
        P = np.zeros((n, n))
        for j in range(n):
            P[j, p[j] - 1] = 1  #Permutation Matrix

        Ep = Vdd**2 * np.trace(A @ P @ Ta @ P.T) # Evaluate Cost Function for this Permutation Matrix

        #Checking if Ep is the best Ep found so far and updating the (Ep_best,p_best)
        if Ep < Ep_best:
            Ep_best = Ep
            p_best = p

    Total_time = time.time()-Start_time #Calculating the total time spend to run the exhaustive search

    print("The best Ep is", Ep_best,"found in",Total_time,"seconds using the Exhaustive search")

    return Ep_best


#---------------------------------------------------SIMULATED ALGORITHM---------------------------------------------------------------#
def Simulated_Annealing(n , A , Ta , E , initial_delta , nr_iterations , ALGORITHMS_RUN_TIME):

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
            #print(f"New best solution value, Ep: {Ep_best} Found at iteration: {i}")
        else:
            
            delta = -math.log(initial_delta) / nr_iterations  #Calculating delta
            Temperature =0.1/(i + 1) #Calculating Temperature
            propability = (math.e)**(-delta/Temperature) #Calculating propability
            
            #Keeping the temp_Best_p with a propability of (propability)
            if random.random() <=propability:
                p_to_check = Temp_Best_p  
        
        if ALGORITHMS_RUN_TIME < time.time()-Start_time:
            print("The best Ep is", Ep_best,"found in",time.time()-Start_time,"seconds using the SA Algorithm")

            #Updating plot lists
            time_running.append(time.time()-Start_time)
            Ep_plot.append(Ep_best)
            Iteration_plot.append(i)

            #Creating plot for visualization of our results
            ax1.plot(Iteration_plot, Ep_plot,'w-')
            ax2 = ax1.twiny()
            ax2.plot(time_running, Ep_plot , '-o')

            plt.title(f'SA ALGORITHM (n={n})')
            ax1.set_xlabel('Number of iterations')
            ax1.set_ylabel('Ep value')
            ax2.set_xlabel('Time running(seconds)')

            
            return time_running , Ep_plot



    print("The best Ep is", Ep_best,"found in",time.time()-Start_time,"seconds using the SA Algorithm")

    #Updating plot lists
    time_running.append(time.time()-Start_time)
    Ep_plot.append(Ep_best)
    Iteration_plot.append(nr_iterations)

    #Creating plot for visualization of our results
    ax1.plot(Iteration_plot, Ep_plot,'w-')
    ax2 = ax1.twiny()
    ax2.plot(time_running, Ep_plot , '-o')

    plt.title(f'SA ALGORITHM (n={n})')
    ax1.set_xlabel('Number of iterations')
    ax1.set_ylabel('Ep value')
    ax2.set_xlabel('Time running(seconds)')

    
    return time_running , Ep_plot


#---------------------------------------------------FANT ALGORITHM--------------------------------------------------------------------#
def FANT(n , A , Ta , E , nr_iterations , ALGORITHMS_RUN_TIME):
   

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

            #print(f"New best solution value, Ep: {Ep} Found at iteration: {no_iteration}")

            best_p = p
            increment = 1
            init_trace(n, increment, trace)  #Re-initialize trace if a better Ep is found
        else:
            update_trace(n, p, best_p, increment, trace)  #Update trace if a better Ep is not found in this iteration

        if ALGORITHMS_RUN_TIME < time.time()-Start_time:
            print("The best Ep is", best_Ep,"found in",time.time()-Start_time,"seconds using the FANT Algorithm")

            #Update plot lists
            time_running.append(time.time()-Start_time)
            Ep_plot.append(best_Ep)
            Iteration_plot.append(no_iteration)

            #Create plot for visualization of our results
            ax1.plot(Iteration_plot, Ep_plot,'w-')
            ax2 = ax1.twiny()
            ax2.plot(time_running, Ep_plot , '-o')

            plt.title(f'FANT ALGORITHM (n={n})')
            ax1.set_xlabel('Number of iterations')
            ax1.set_ylabel('Ep value')
            ax2.set_xlabel('Time running(seconds)')

            return time_running , Ep_plot


    print("The best Ep is", best_Ep,"found in",time.time()-Start_time,"seconds using the FANT Algorithm")

    #Update plot lists
    time_running.append(time.time()-Start_time)
    Ep_plot.append(best_Ep)
    Iteration_plot.append(nr_iterations)

    #Create plot for visualization of our results
    ax1.plot(Iteration_plot, Ep_plot,'w-')
    ax2 = ax1.twiny()
    ax2.plot(time_running, Ep_plot , '-o')

    plt.title(f'FANT ALGORITHM (n={n})')
    ax1.set_xlabel('Number of iterations')
    ax1.set_ylabel('Ep value')
    ax2.set_xlabel('Time running(seconds)')

    return time_running , Ep_plot


#---------------------------------------------------TABU ALGORITHM---------------------------------------------------------------------#

def TABU(n, A , Ta, E, nr_iterations , nr_of_trials , Aspiration , Tabu_duration , ALGORITHMS_RUN_TIME):

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
                        #print(f"Solution of value: {Temp_Ep} found at iteration: {current_iteration} and resolution: {nr_resolutions}")

                    for i in range(n-1):
                        for j in range(i+1, n):
                            if i != i_retained and i != j_retained and j != i_retained and j != j_retained:
                                delta[i][j] = compute_delta_part(A, Ta, Temp_p, delta, i, j, i_retained, j_retained)
                            else:
                                delta[i][j] = compute_delta(n, A, Ta, Temp_p, i, j)

                if ALGORITHMS_RUN_TIME < time.time()-Start_time:
                    print("The best Ep is", Best_Ep,"found in",time.time()-Start_time,"seconds using the TABU Algorithm")

                    #Updating plot lists
                    time_running.append(time.time()-Start_time)
                    Ep_plot.append(Best_Ep)
                    Resolutions_plot.append(nr_resolutions)
                    
                    #Creating plot for visualization of our results
                    ax1.plot(Resolutions_plot, Ep_plot,'w-')
                    ax2 = ax1.twiny()
                    ax2.plot(time_running , Ep_plot , '-o')

                    plt.title(f'TABU ALGORITHM (n={n})')
                    ax1.set_xlabel('Number of resolutions')
                    ax1.set_ylabel('Ep value')
                    ax2.set_xlabel('Time running(seconds)')

                    return time_running,Ep_plot

        if ALGORITHMS_RUN_TIME < time.time()-Start_time:
            print("The best Ep is", Best_Ep,"found in",time.time()-Start_time,"seconds using the TABU Algorithm")

            #Updating plot lists
            time_running.append(time.time()-Start_time)
            Ep_plot.append(Best_Ep)
            Resolutions_plot.append(nr_resolutions)
            
            #Creating plot for visualization of our results
            ax1.plot(Resolutions_plot, Ep_plot,'w-')
            ax2 = ax1.twiny()
            ax2.plot(time_running , Ep_plot , '-o')

            plt.title(f'TABU ALGORITHM (n={n})')
            ax1.set_xlabel('Number of resolutions')
            ax1.set_ylabel('Ep value')
            ax2.set_xlabel('Time running(seconds)')

            return time_running,Ep_plot
            



    print("The best Ep is", Best_Ep,"found in",time.time()-Start_time,"seconds using the TABU Algorithm")

    #Updating plot lists
    time_running.append(time.time()-Start_time)
    Ep_plot.append(Best_Ep)
    Resolutions_plot.append(nr_of_trials)
    
    #Creating plot for visualization of our results
    ax1.plot(Resolutions_plot, Ep_plot,'w-')
    ax2 = ax1.twiny()
    ax2.plot(time_running, Ep_plot , '-o')

    plt.title(f'TABU ALGORITHM (n={n})')
    ax1.set_xlabel('Number of resolutions')
    ax1.set_ylabel('Ep value')
    ax2.set_xlabel('Time running(seconds)')

    return time_running,Ep_plot


#---------------------------------------------------ILSA ALGORITHM---------------------------------------------------------------------#
def ILSA(n , A, Ta , E , nr_iterations , SHUFFLE_TOLERANCE , propability_of_change , Nr_Swaps_LI , ALGORITHMS_RUN_TIME):

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

            #print(f"New best solution value, temp_Ep: {temp_Ep} Found at iteration: {i}")
            count = 0
   
        else:
            #If a better solution is not found for a number of iterations(SHUFFLE_TOLERANCE) then randomly shuffle our current p
            count += 1
            if count > SHUFFLE_TOLERANCE:
                random.shuffle(temp_p)
                count = 0
        
        if ALGORITHMS_RUN_TIME < time.time()-Start_time:
            print("The best Ep is", Best_Ep,"found in",time.time()-Start_time,"seconds using the ILSA Algorithm")

            #Update plot lists 
            time_running.append(time.time()-Start_time)
            Ep_plot.append(Best_Ep)
            Iteration_plot.append(i)
            

            #Create plot to visualize our results
            ax1.plot(Iteration_plot, Ep_plot,'w-')
            ax2 = ax1.twiny()
            ax2.plot(time_running, Ep_plot , '-o')

            plt.title(f'ILSA ALGORITHM (n={n})')
            ax1.set_xlabel('Number of iterations')
            ax1.set_ylabel('Ep value')
            ax2.set_xlabel('Time running(seconds)')

            return time_running , Ep_plot
                


    print("The best Ep is", Best_Ep,"found in",time.time()-Start_time,"seconds using the ILSA Algorithm")

    #Update plot lists 
    time_running.append(time.time()-Start_time)
    Ep_plot.append(Best_Ep)
    Iteration_plot.append(nr_iterations)
    

    #Create plot to visualize our results
    ax1.plot(Iteration_plot, Ep_plot,'w-')
    ax2 = ax1.twiny()
    ax2.plot(time_running, Ep_plot , '-o')

    plt.title(f'ILSA ALGORITHM (n={n})')
    ax1.set_xlabel('Number of iterations')
    ax1.set_ylabel('Ep value')
    ax2.set_xlabel('Time running(seconds)')

    return time_running , Ep_plot
        

#---------------------------------------------------GRASP ALGORITHM---------------------------------------------------------------------#
def Grasp(n , A , Ta , E , iterations , beta , alpha , ALGORITHMS_RUN_TIME):

    #---------------------------------------------------------FUNCTIONS-----------------------------------------------------------#
    
    
    #Creates a list with the best pairs using the Ta values and beta
    def Construct_best_pairs_list_for_all_elements(n , Ta , beta):
        Ta=-Ta #Using the -Ta matrix because when i calculate the E i multiply Ta with A that has negative values and i want to keep the lowest values of this
        List_Of_Best_Moves = [] #Initialize list of best moves
        for i in range(n):
            for j in range(n):
                if j!=i:
                    List_Of_Best_Moves.append([Ta[i][j],i,j])   #Creating a list with sublist that have every value of Ta and their index i,j except the elements of the diagonal
        List_Of_Best_Moves = sorted(List_Of_Best_Moves, key=lambda x: x[0]) #Sort the sublists of the list according to the first element of every sublist
        del List_Of_Best_Moves[int(-beta*(n**2-n)):] #Removing (int(-beta*(n**2-n))) number of sublists from the end of the list
        return List_Of_Best_Moves

    #Function use for the constraction phase 2 where it finds the next index based on the Ta values for the moves from our current index to all the possible others existing in candidates list
    def get_values_for_possible_next_move(Ta, candidates_list , current_index ,alpha):
                Ta = -Ta   #Using the -Ta matrix because when i calculate the E i multiply Ta with A that has negative values and i want to keep the lowest values of this
                List_of_possible_moves_from_current_index = [] #Initialize List of possible moves from our current index

                #Filling the List of possible moves using the candidates list
                for a in range(n):
                    if a in candidates_list:
                        List_of_possible_moves_from_current_index.append([Ta[a][current_index],a]) #Has sublist that have the Ta value for moving from our current index to an index(a) and the index a

                List_of_possible_moves_from_current_index = sorted(List_of_possible_moves_from_current_index, key=lambda x: x[0]) #Sort the sublists of the list according to the first element of every sublist

                if int(alpha*(n))!=len(List_of_possible_moves_from_current_index):   
                    del List_of_possible_moves_from_current_index[int(-alpha*(len(List_of_possible_moves_from_current_index))):] #Deleting int(-alpha*(len(List_of_possible_moves_from_current_index)) amount of sublist from the end

                List_of_possible_moves_from_current_index= random.choice(List_of_possible_moves_from_current_index) #Randomly chosing one sublist
                next_index = List_of_possible_moves_from_current_index[1] #Getting the next index
            
                return next_index

    #Function for Constraction phase 1 where a starting pair is selected for our permutation p
    def Constraction_Phase_1(List_Of_Best_Moves ,p):
        Starting_Selected_Pair = random.choice(List_Of_Best_Moves) #Randomly selecting the starting pair from the List_of_Best_Moves
        #Updating p
        p.append(Starting_Selected_Pair[1])
        p.append(Starting_Selected_Pair[2])
        return p

    #Function for Constraction Phase 2 where we taking the starting pair and keep adding the next element in p until there is no possible element left to be added
    def Constraction_Phase_2(p, Ta , candidates_list ,alpha):
        current_index = p[-1] #Initiliazing the place we are in the permutation p

        #Adding elements to p one by one
        for i in range(n-3):
            next_index = get_values_for_possible_next_move(Ta,candidates_list , current_index , alpha) #Getting the next index

            p.append(next_index) #Updating p
            candidates_list.remove(next_index) #Updating candidates list
            current_index = next_index #Updating current index (moving from our current index to the next index we chose)

        p.append(candidates_list[0]) #No need to use function for next index as there is only one left we just add it to the p
        return p
    
    #Function for local search of best solution for our current permutation p
    def Local_Search(n, p , A ,Ta, Best_Ep):

        #Function for checking if temp_Ep is better than our Best Ep and updating the Best Ep and Best p value
        def Check_If_Best_Ep(temp_Ep ,temp_p, Best_Ep):
            if temp_Ep < Best_Ep:
                #Update best ep,best p
                Best_Ep = temp_Ep
                Best_p = temp_p
                #Updating plot lists
                time_running.append(time.time()-Start_time)
                Ep_plot.append(Best_Ep)
                Iteration_plot.append(iteration)

                #print(f"New best solution value, Ep: {Best_Ep} Found at iteration: {iteration}") 
            return Best_Ep
        
        #CALCULATE EP FOR A GIVEN PERMUTATION p 
        def calculate_cost(n , p , A , Ta):    
            #CALCULATING PERMUTATION MATRIX BASED ON SPECIFIC PERMUTATION pe
            P = np.zeros((n, n))
            for j in range(n):
                P[j, p[j] - 1] = 1  # Permutation Matrix

            Ep = Vdd**2 * np.trace(A @ P @ Ta @ P.T)  # Evaluate Cost Function for this Permutation Matrix
            return Ep,p

        #Cheking if our current p gives a best ep before we start the local search
        temp_Ep , temp_p  = calculate_cost(n, p , A ,Ta)
        Best_Ep = Check_If_Best_Ep(temp_Ep ,temp_p, Best_Ep)

        #Local search for neighbourhood of 3
        for i in range(n-2):
            #For a neighbourhood starting as [1,2,3](we dont check for [1,3,2] because it will be checked in the next iteration and for [1,2,3] because is the initial p that is calculated outside of the loop)
            #Checking [3,1,2]
            p_local_Search = p #Creating temp p for Local search
            p_local_Search[i] , p_local_Search[i+1] , p_local_Search[i+2]  = p[i+2] , p[i] , p[i+1] #Swap certain elements for local search
            temp_Ep , temp_p  = calculate_cost(n , p_local_Search , A , Ta ) #Calculate cost of the p local search created
            Best_Ep = Check_If_Best_Ep(temp_Ep ,temp_p, Best_Ep) #Check if cost coming from local search is best and if so update values

            #Checking [3,2,1]
            p_local_Search = p #Creating temp p for Local search
            p_local_Search[i] , p_local_Search[i+1] , p_local_Search[i+2]  = p[i+2] , p[i+1] , p[i]  #Swap certain elements for local search
            temp_Ep , temp_p  = calculate_cost(n , p_local_Search , A , Ta ) #Calculate cost of the p local search created
            Best_Ep = Check_If_Best_Ep(temp_Ep ,temp_p, Best_Ep) #Check if cost coming from local search is best and if so update values

            #Checking [2,1,3]
            p_local_Search = p #Creating temp p for Local search
            p_local_Search[i] , p_local_Search[i+1] , p_local_Search[i+2]  = p[i+1] , p[i] , p[i+2] #Swap certain elements for local search
            temp_Ep , temp_p  = calculate_cost(n , p_local_Search , A , Ta ) #Calculate cost of the p local search created
            Best_Ep = Check_If_Best_Ep(temp_Ep ,temp_p, Best_Ep) #Check if cost coming from local search is best and if so update values

            #Checking [2,3,1]
            p_local_Search = p  #Creating temp p for Local search
            p_local_Search[i] , p_local_Search[i+1] , p_local_Search[i+2] = p[i+1] , p[i+2] , p[i] #Swap certain elements for local search
            temp_Ep , temp_p  = calculate_cost(n , p_local_Search , A , Ta ) #Calculate cost of the p local search created
            Best_Ep = Check_If_Best_Ep(temp_Ep ,temp_p, Best_Ep) #Check if cost coming from local search is best and if so update values
        return Best_Ep

 
        
    
    #-----------------------------------------------------INITIALIZATION-----------------------------------------------------------------------------------#
    List_Of_Best_Moves = Construct_best_pairs_list_for_all_elements(n, Ta ,beta)  #Creating the list of best moves (once) that will be used through our algorithm
    
    #Initialize Best values
    Best_p = list(range(n))
    Best_Ep = E

    #Initilize plot lists
    Start_time=time.time()
    time_running=[0,0]
    Ep_plot = [E,E]
    Iteration_plot = [0,0]
    fig, ax1 = plt.subplots() #Create plot
    #--------------------------------------------------GRASP MAIN--------------------------------------------------------------------------------------# 

    for iteration in range(iterations):

        p=[] #Initilize p

        #In Stage 1 of the construction phase, we select the starting pair and adding them in p
        p = Constraction_Phase_1(List_Of_Best_Moves , p)

        
        candidates_list = list(range(n)) #Initialize candidates list
        candidates_list.remove(p[0]) #Update candidates list according to starting pair
        candidates_list.remove(p[1]) #Update candidates list according to starting pair

        #In Stage 2 of the construction phase we assign elements to our permutation one by one
        p = Constraction_Phase_2(p, Ta , candidates_list ,alpha)

        #Local Search Stage-Checking for Best_EP
        Best_Ep = Local_Search(n , p , A ,Ta , Best_Ep)

        if ALGORITHMS_RUN_TIME < time.time()-Start_time:
            print("The best Ep is", Best_Ep,"found in",time.time()-Start_time,"seconds using the GRASP Algorithm")

            #Updating plot lists
            time_running.append(time.time()-Start_time)
            Ep_plot.append(Best_Ep)
            Iteration_plot.append(iteration)    
            
            #Create plot for the visualization of our results
            ax1.plot(Iteration_plot, Ep_plot,'w-')
            ax2 = ax1.twiny()
            ax2.plot(time_running, Ep_plot ,'-o' )

            plt.title(f'GRASP ALGORITHM (n={n})')
            ax1.set_xlabel('Number of iterations')
            ax1.set_ylabel('Ep value')
            ax2.set_xlabel('Time running(seconds)')

        
            return time_running , Ep_plot


    
    print("The best Ep is", Best_Ep,"found in",time.time()-Start_time,"seconds using the GRASP Algorithm")

    #Updating plot lists
    time_running.append(time.time()-Start_time)
    Ep_plot.append(Best_Ep)
    Iteration_plot.append(iterations)    
   
    #Create plot for the visualization of our results
    ax1.plot(Iteration_plot, Ep_plot,'w-')
    ax2 = ax1.twiny()
    ax2.plot(time_running, Ep_plot , '-o')

    plt.title(f'GRASP ALGORITHM (n={n})')
    ax1.set_xlabel('Number of iterations')
    ax1.set_ylabel('Ep value')
    ax2.set_xlabel('Time running(seconds)')

   
    return time_running , Ep_plot



#---------------------------------LIST WITH VALUES FOR EACH ALGORITHM-------------------------------------------------------------------#
ALGORITHMS_RUN_TIME = 10  #In seconds
#Best_Ep_Exhaustive_search = exhaustive_search(n,A,Ta,E)             
#If u want to use and compare your algoritms based on the algorithm run time just put a high enough iteration number for every algorithm(resolutions number for tabu)      
#If u care about time according to number of iteration just put a high enough algorithm run time value

Time_Running_ILSA , Ep_plot_ILSA =ILSA(n,  A , Ta , E , 10000000000 , 100 , 0.5 , 3 , ALGORITHMS_RUN_TIME )   

Time_Running_FANT , Ep_plot_FANT = FANT(n , A , Ta , E , 1000000000 , ALGORITHMS_RUN_TIME)    

Time_Running_TABU , Ep_plot_TABU = TABU(n , A , Ta , E , 15000 , 100000000 , 5*n *n , 8*n , ALGORITHMS_RUN_TIME)

Time_Running_SA , Ep_plot_SA = Simulated_Annealing(n , A , Ta , E , 0.00001 , 20000000000 , ALGORITHMS_RUN_TIME )  
    
Time_Running_GRASP , Ep_plot_GRASP = Grasp(n , A , Ta , E , 1000000000 , 0.5 , 0.5 , ALGORITHMS_RUN_TIME)


#------------------------------------------------------------PLOTING ALL ALGORITHMS FOR COMPARISON-------------------------------------#
fig,ax1 = plt.subplots()

#plt.axhline(y=Best_Ep_Exhaustive_search, color='gray', linestyle='--', label=f'y = {Best_Ep_Exhaustive_search}')

ax1.plot(Time_Running_ILSA , Ep_plot_ILSA ,label='ILSA')
ax1.plot(Time_Running_TABU, Ep_plot_TABU ,label = 'TABU')
ax1.plot(Time_Running_SA , Ep_plot_SA ,label = 'SA')
ax1.plot(Time_Running_FANT , Ep_plot_FANT ,label='FANT')
ax1.plot(Time_Running_GRASP , Ep_plot_GRASP ,label='GRASP')



plt.title(f"COMPARING ALGORITHMS (n={n})")
ax1.set_xlabel("Time running(seconds)")
ax1.set_ylabel("Ep value")

plt.legend()
plt.show()

