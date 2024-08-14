import numpy as np
from itertools import permutations
import time
import matplotlib.pyplot as plt


Total_time_spend_list=[]   #INITIALIZE THE LIST THAT HOLDS THE TIME SPEND FOR EXHAUSTIVE SEARCH TO FIND THE SOLUTION FOR EACH TRY
Best_ep_list = []          #INITIALIZE THE LIST THAT HOLDS THE BEST EP VALE FOUND FROM EXHAUSTIVE SEARCH FOR EACH TRY

Nr_of_trials =10           #NUMBER OF TRIES OF EXHAUSTIVE SEARCH

for a in range(Nr_of_trials):

    #INITIALIZE THE PARAMETERS USED
    Vdd = 1
    CL = 1
    Lambda = 3
    m = 10000   #(nxm) is the size of the data matrix (n = number of lines),(m = number of periods)
    n = 3

    #-------------------------------------------------CALCULATING (A,R0,R1,TA,E)---------------------------------------------------------#
    
    #Calculate A matrix
    h0 = np.eye(n)     # Create h0 matrix (identity matrix)
    
    h1 = np.diag(np.ones(n-1), k=1) + np.diag(np.ones(n-1), k=-1)    # Create h1 
    h1 += np.diag(np.zeros(n))

    A = ((1 + 2 * Lambda) * h0 - Lambda * h1) * CL  #Create A matrix

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



    #------------------------------EXHAUSTIVE SEARCH ALGORITHM-------------------------------------------------------#
    def exhaustive_search(n,A,Ta,E):
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

        return Total_time,Ep_best


    Total_time , Ep_best = exhaustive_search( n , A , Ta , E )  #Calling the Exhaustive Search Function
    
    #Updating the Total time spend list and Best ep list
    Total_time_spend_list.append(Total_time)
    Best_ep_list.append(Ep_best)


#Calculating Average time and average best Ep found
Average_time = sum(Total_time_spend_list)/len(Total_time_spend_list)
Average_Ep = sum (Best_ep_list) / len(Best_ep_list)

#Creating diagram to visualize our results
plt.scatter( Total_time_spend_list, Best_ep_list)
plt.plot(Average_time, Average_Ep, 'rx ') 
plt.xlabel('Total time(seconds)')
plt.ylabel('Best Ep value')
plt.title(f"Trying {Nr_of_trials} times the Exhaustive Search for n={n}\n(The average best Ep is {format(Average_Ep,'.4f')} and the average time of calulation is {format(Average_time,'.4f')} seconds) ) ")
plt.show()



#---------------------------------------------------GENERAL COMMENTS----------------------------------------------#
#THE EXHAUSTIVE SEARCH FUNCTION IS USED FOR n<=11.At n=11 the device has a bit of trouble calculating the result but it manages to do so after 10 minutes
#At n = 12 the device fails to work because of the vast amount of memory needed for the exhaustive search function