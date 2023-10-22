import numpy as np
from model import *
from population import patch, metaPopulation
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({'font.size': 25})




#Define the epsilon's to be evaluated
emin = 10e-5
emax = 0.1
epsilons = np.linspace(emin, emax, 10)




M = np.array([[-1, 1], [1, -1]])


# Number of iterations
iter = 1000
counter = 0

    
#plot definition
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_yscale('log', base = 10)
ax.set_xscale('log', base = 10)
ax.set_ylim([10e-3, 1])




param_file = open("params.txt", "w")

error_list  = []
slopes = []
first = []
for i in range(iter):
    print('----------------------------------------Iteration: ', i)
    neutral_beta = np.random.uniform(0.6, 4, size = 2)
    neutral_gamma = np.random.uniform(0, 0.5, size = 2)
    neutral_r = 0.5 - neutral_gamma
    neutral_k = np.random.uniform(0.1, 8, size = 2)
    print(neutral_gamma + neutral_r)
    R0 = neutral_beta/(neutral_gamma + neutral_r)
    
    TSstar = 1/R0

    TTstar = 1 - TSstar
    TIstar = TTstar/(1 + neutral_k*(R0 - 1))
    TDstar = TTstar - TIstar

    initial_conditions1 = np.array([TSstar[0], 0.5*TIstar[0], 0.5*TIstar[0], 0.25*TDstar[0], 0.25*TDstar[0], 0.25*TDstar[0], 0.25*TDstar[0]]) + np.random.normal(0, 0.01, size = 7)
    initial_conditions2 = np.array([TSstar[1], 0.5*TIstar[1], 0.5*TIstar[1], 0.25*TDstar[1], 0.25*TDstar[1], 0.25*TDstar[1], 0.25*TDstar[1]]) + np.random.normal(0, 0.01, size = 7) 
    initial_conditions1 = initial_conditions1/np.sum(initial_conditions1)
    initial_conditions2 = initial_conditions2/np.sum(initial_conditions2)






    alpha = np.random.uniform(-1, 1, size = 4)
    

    print('R0 = ', R0)
    print('Sum of initial conditions in A: ', sum(initial_conditions1))
    print('Sum of initial conditions in B: ', sum(initial_conditions2))

    
    if np.all(R0 > 1) and sum(initial_conditions1) == 1.0 and sum(initial_conditions2) == 1.0:
        counter += 1
        
        
        param_file.write("-------------------------------- Iteration  " + str(counter) +"------------- \n")
        param_file.write("r: " + str(neutral_r) + "\n")
        param_file.write("beta: " +str(neutral_beta)+ "\n")
        param_file.write("gamma: "  +str(neutral_gamma)+ "\n")
        param_file.write("k: " + str(neutral_k)+ "\n")
        param_file.write("alpha: " + str(alpha)+ "\n")
        
         
        errors = []
        
        for e in epsilons:
            print("------ Epsilon:", e)
            t = np.linspace(0, 10e8, int(1/e) + 1)
            d = e
            patch1 = patch('A', initial_conditions1, neutral_r[0], neutral_beta[0], neutral_gamma[0], neutral_k[0], d)
            #patch1.define_beta(b)
            patch1.define_K(alpha)
            patch2 = patch('B', initial_conditions2, neutral_r[1], neutral_beta[1], neutral_gamma[1], neutral_k[1], d)
            #patch2.define_beta(b)
            patch2.define_K(alpha)
            patches = [patch1, patch2]
            metapop = metaPopulation(patches, d, M)
            measures = metapop.measures(t)

            if measures['sucess']:

                errors.append(measures['error'])
            else:
                print(">>>>This step was unsuccessful.")
                errors.append(0)
                
        
        
        if  np.all(np.diff(errors) > 0):
            error_list.append(errors)
            slopes.append((np.log10(errors[-1]) - np.log10(errors[0]))/(np.log10(epsilons[-1]) - np.log10(epsilons[0])))
        print(np.diff(np.diff(errors))/np.diff(epsilons[:-1]))

        first.append(np.log10(errors[0]))

mean, stdev = np.mean(slopes), np.std(slopes)



n = 0
for i in range(len(error_list)):
    z = np.abs((slopes[i] - mean)/stdev)
    print("z score: ", z)
    #if np.all(mean_grad_error[i]) <= np.all(mean) + 0.1*stdev and np.all(mean_grad_error[i]) >= np.all(mean) - 0.1*stdev and np.all(np.diff(error_list[i]) > 0):
    if np.all(z < 1) and np.all(slopes[i] > 0):
        ax.plot(epsilons, error_list[i])
        n += 1

print("the mean slope is:", mean)


plt.title('Errors in the replicator approximation of ' + str(n) + ' parameters sets')
plt.xlabel('$\epsilon$')
plt.ylabel('Error')
ax.invert_xaxis()


plt.legend()
plt.show()

param_file.write("-------------------------------------------------------------- \n")
param_file.write("Mean gradient of error: " + str(mean))
param_file.write("\nStandard deviation of gradient of error: " + str(stdev))
param_file.close()





