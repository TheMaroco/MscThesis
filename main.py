import numpy as np
from model import *
from population import patch, metaPopulation


t = np.linspace(0, 10e8)
# initial_conditions1 = np.random.rand(6)
# initial_conditions1 = initial_conditions1/(np.sum(initial_conditions1)/(1 - (1 + 2)/6))
# initial_conditions2 = np.random.rand(6)
# initial_conditions2 = initial_conditions2/(np.sum(initial_conditions2)/(1 - (1 + 2)/5))

# initial_conditions1 = np.concatenate([np.array([0.5]), initial_conditions1])
# initial_conditions2 = np.concatenate([np.array([0.5]), initial_conditions2])

# >>> Using the random initial conditions it seems that the error isn't bounded by sqrt(epsilon). Also this bound is only valid for d in the scale of epsilon. I guess this makes since 
# we only have the discrete replicator for the slow movement cases. For the fast movement case we know from Thao that the approximation is done by the mean field replicator and it doesn't make sense 
# to discretize this equation (it's 1D).



initial_conditions1 = np.array([0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25]) + np.random.uniform(0.2, 0.5, size = 7)
initial_conditions2 = np.array([0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25]) + np.random.uniform(0.2, 0.5, size = 7) 
initial_conditions1 = initial_conditions1/np.sum(initial_conditions1)
initial_conditions2 = initial_conditions2/np.sum(initial_conditions2)
    

neutral_r = 0.1
neutral_beta = 6

epsilon = 0.1
d = epsilon
M = np.array([[-1, 1], [1, -1]])
patch1 = patch('A', initial_conditions1, neutral_r, neutralbeta = 4, neutralgamma = 2, neutralk = 3, epsilon = epsilon)
b = [.2, .3]
patch1.define_beta(b) 

alpha = [0.1, 0.4, 0.3, 0.1]
#patch1.define_K(alpha)

patch2 = patch('B', initial_conditions2, neutral_r, neutralbeta = 5, neutralgamma = 2, neutralk = 2, epsilon = epsilon)
b = [0.1, 0.6]
patch2.define_beta(b)
alpha = [.1, .3, .5, .1]
patch2.define_K(alpha)


patch1.describe()
patch2.describe()

patches = [patch1, patch2]
metapop = metaPopulation(patches, d, M)
print('Patch A R0:', patch1.R0)
print('Patch B R0:', patch2.R0)
Alambdas = patch1.invasion_fitness()
Blambdas = patch2.invasion_fitness()
print('Lambdas for patch A:', Alambdas)
print('Lambdas for patch B:', Blambdas)
print('Average R0', metapop.meanR0())
print('Average lambda1_2', metapop.avgInvasionfitness()[0])
print('Average lambda2_1', metapop.avgInvasionfitness()[1])
print('w:', metapop.measures(t)['w'])

print('error in the approximation:', metapop.measures(t)['error'])


plot(metapop.measures(t), t)
plt.show()

simplePlot(metapop.measures(t), t[:2000])
plt.show()

tau = epsilon*np.linspace(0, t[-1]/epsilon, 100)
mean_replicator = metapop.mean_replicator(epsilon*tau, metapop.measures(t)['avg_z1'][0])
avg_replicator = metapop.measures(t)['avg_replicator']

plt.ylim([0, 1])
plt.plot(t, mean_replicator, label = 'mean replicator')
plt.plot(t, avg_replicator, label = 'average replicator')
plt.plot(t, metapop.measures(t)['avg_z1'], label = '$\overline{z}_1$')
plt.title('Mean replicator vs. Average Replicator')
plt.legend()
plt.show()




errors = []
Serrors = []
Ierrors = []
Derrors = []

alpha = [0.1, 0.2, 0.3, 0.2]


epsilons = np.linspace(0.001, 0.4, 25)
for e in epsilons:
    patch1 = patch('A', initial_conditions1, 1, 6, 2, 0.5, e)
    #patch1.define_beta(b)
    patch1.define_K(alpha)
    patch2 = patch('B', initial_conditions2, 1, 5, 2, 0.5, e)
    #patch2.define_beta(b)
    patch2.define_K(alpha)
    patches = [patch1, patch2]
    metapop = metaPopulation(patches, d, M)
    measures = metapop.measures(t)
    
    errors.append(measures['error'])
    Serrors.append(measures['Serror'])
    Ierrors.append(measures['Ierror'])
    Derrors.append(measures['Derror'])



plt.plot(epsilons, errors, label = 'Total error ')
plt.plot(epsilons, np.sqrt(epsilons), label = '$\sqrt{\epsilon}$')
plt.plot(epsilons, Serrors, label = 'Error in S')
plt.plot(epsilons, Ierrors, label = 'Error in I')
plt.plot(epsilons, Derrors, label = 'Error in D')
plt.title('Errors in the replicator approximation')

plt.legend()
plt.show()


 

