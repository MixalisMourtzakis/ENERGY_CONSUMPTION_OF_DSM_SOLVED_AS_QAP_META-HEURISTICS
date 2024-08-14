import numpy as np
from itertools import permutations
import time
import random
import matplotlib.pyplot as plt
import pandas as pd


#INITIALIZE THE PARAMETERS USED
Vdd = 1
CL = 1
Lambda = 3
m=10000   #(nxm) is the size of the data matrix (n = number of lines),(m = number of periods)
n=10

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




#--------------------------------------------GRASP ALGORITHM---------------------------------------------------------------------#



def Grasp(n , A , Ta , E , iterations , beta ,alpha):

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

                print(f"New best solution value, Ep: {Best_Ep} Found at iteration: {iteration}") 
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
    
    #Updating plot lists
    time_running.append(time.time()-Start_time)
    Ep_plot.append(Best_Ep)
    Iteration_plot.append(iterations)

    print("The best Ep is", Best_Ep,"found in",time.time()-Start_time,"seconds using the GRASP Algorithm")    
    
    #Create plot for the visualization of our results
    ax1.plot(time_running, Ep_plot,'w-')
    ax2 = ax1.twiny()
    ax2.plot(Iteration_plot, Ep_plot , '-o')

    plt.title('GRASP ALGORITHM')
    ax1.set_xlabel('Time running(seconds)')
    ax1.set_ylabel('Ep value')
    ax2.set_xlabel('Number of iterations')

   
    plt.show()
    return time_running , Ep_plot




Grasp(n,A,Ta,E,1000,0.5,0.5) 



#####GENERAL COMMENTS#######

#The bigger the beta i chose the less choices i have from the List_of_Best_moves