import numpy as np
from model import *
from population import patch, metaPopulation


t = np.linspace(0, 1000)

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

simplePlot(metapop.measures(t), t[:2000])
plt.show()






 

