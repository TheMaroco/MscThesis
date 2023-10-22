import numpy as np
from scipy import integrate
from model import analysis, system, neutralsystem, solve, replicator1
import matplotlib.pyplot  as  plt




class patch:

    def __init__(self, name, v0, r, neutralbeta, neutralgamma, neutralk, epsilon):
        self.name = name
        self.v0 = np.array(v0)
        self.v = np.array(v0)
        self.r = np.array(r)
        self.nbeta = np.array(neutralbeta)
        self.beta = self.nbeta*np.ones(2)
        self.ngamma = np.array(neutralgamma)
        self.sgamma = self.ngamma*np.ones(2)
        self.cgamma = self.ngamma*np.ones(4)
        self.nk = np.array(neutralk)
        self.K = self.nk*np.ones(4)
        self.p = self.q = 0.5
        self.epsilon = epsilon
        self.m = np.array(r) + np.array(neutralgamma)
        self.R0 = np.array(neutralbeta)/(np.array(r) + np.array(neutralgamma))
        self.nSstar = 1/self.R0
        self.nTstar = 1 - self.nSstar
        self.nIstar = self.nTstar/(1 + self.nk*(self.R0 -1))
        self.nDstar = self.nTstar - self.nIstar
        self.mu = self.nIstar/self.nDstar
        self.detP = np.linalg.det(np.array([[2*self.nTstar, self.nIstar], [self.nDstar, self.nTstar]]))
        self.theta1 = 2*self.nbeta*self.nSstar*self.nTstar**2/self.detP
        self.theta2 = self.ngamma*self.nIstar*(self.nIstar + self.nTstar)/self.detP
        self.theta3 = self.ngamma*self.nTstar*self.nDstar/self.detP
        self.theta4 = 2*self.m*self.nTstar*self.nDstar/self.detP
        self.theta5 = self.nbeta*self.nTstar*self.nIstar*self.nDstar/self.detP
        self.theta = self.theta1 + self.theta2 + self.theta3 + self.theta4 + self.theta5
        self.lambda1_2 = 0
        self.lambda2_1 = 0
    
    def define_beta(self, b = [0, 0]):
        """Function to generate the non-neutral betas. bi is the strain specific reproduction rates."""
        #To generalize this to more strains -> make bi be a higher dimensional vector, use a for (over i in range(len(bi))) and append with the formula nbeta*(1 + epsilon*bi[i])
        self.beta = np.array([self.nbeta*(1 + self.epsilon*b[0]), self.nbeta*(1 +  self.epsilon*b[1])])
        #self.b = np.array([self.nbeta + self.epsilon*b[0], self.nbeta + self.epsilon*b[1]])
        return self.beta
    def define_sgamma(self, gammai = [0, 0]):
        """Function to generate the non-neutral single infectio gammas. gammai is the strain specific clearance rates."""
        self.sgamma = np.array([self.ngamma*(1 + self.epsilon*gammai[0]), self.ngamma*(1 + self.epsilon*gammai[1])])
        return self.sgamma

    def define_cgamma(self, cgammai = [0, 0, 0, 0]):
        self.cgamma = self.ngamma*np.array([1 + self.epsilon*cgammai[0], 1 + self.epsilon*cgammai[1], 1 + self.epsilon*cgammai[2], 1 + self.epsilon*cgammai[3]])
        return self.cgamma
    #Keep doing this for the remaining parameters
    
    def define_K(self, alpha = np.zeros(4)):
        self.K = np.array([self.nk + self.epsilon*alpha[0], self.nk + self.epsilon*alpha[1], self.nk + self.epsilon*alpha[2], self.nk + self.epsilon*alpha[3] ])
        return self.K

    def define_p(self, p = 0):  
        self.p = 0.5 + self.epsilon*p
        return self.p

    def define_q(self, q = 0):
        self.q = 0.5 + self.epsilon*q
        return self.q

    def invasion_fitness(self):
        self.lambda1_2 = self.theta1*(self.beta[0] - self.beta[1]) + self.theta2*(-self.sgamma[0] + self.sgamma[1]) + self.theta3*(-self.cgamma[1] - self.cgamma[2] + 2*self.cgamma[3]) + self.theta4*(self.q - self.p) + self.theta5*(self.mu*(self.K[2] - self.K[1]) + self.K[2] - self.K[3])
        self.lambda2_1 = self.theta1*(self.beta[1] - self.beta[0]) + self.theta2*(-self.sgamma[1] + self.sgamma[0]) + self.theta3*(-self.cgamma[2] - self.cgamma[1] + 2*self.cgamma[0]) + self.theta4*(self.p - self.q) + self.theta5*(self.mu*(self.K[1] - self.K[2]) + self.K[1] - self.K[0])
        return self.lambda1_2, self.lambda2_1

    def neutralmodel(self, t):
        return integrate.odeint(neutralsystem, self.v0, t, args = (self.r, self.nbeta, self.ngamma,  self.nk, self.p, self.q, 0))



    
    def describe(self, option = 'parameters'):
        """Function to describe the patch. Option decides things to be described."""
        print('---------------------------------------------------------------------------')
        if option == 'parameters':
            print('Summary of patch', self.name, 'parameteres:')
            print('Strain reproduction rates:', self.beta)
            print('Single infection clearance rates:', self.sgamma)
            print('Double infection clearance rates:', self.cgamma)
            print('Altered susceptibilities:', self.K)
            print('Probabiltity that second strain is transmited in co-infection:', self.q)
            print('Probabiltity that first strain is transmited in co-infection:', self.p)
            print('Patch', self.name, 'R0:', self.R0)
            print('Patch ', self.name, 'theta:', self.theta)


        if option == 'neutral equilibria':
            print('Patch', self.name, 'neutral S*:', self.nSstar)
            print('Patch', self.name, 'neutral T*:', self.nTstar)
            print('Patch', self.name, 'neutral I*:', self.nIstar)
            print('Patch', self.name, 'neutral D*:', self.nDstar)
        
        if option == 'invasion fitness':
            print('Invasion fitness of 1 -> 2:', self.lambda1_2)
            print('Invasion fitness of 2 -> 1:', self.lambda2_1)

        print('---------------------------------------------------------------------------')
        pass



class metaPopulation:
    def __init__(self, patches, d, M):
        n = len(patches) 
        v0s = []
        rs = np.zeros(n)
        nbetas  =  np.zeros(n)
        ngammas =  np.zeros(n)
        nks = n*[None]
        
        for i in range(7):
            for patch in patches:
                v0s.append(patch.v0[i])
                
        for i in range(n):
            rs[i] = patches[i].r
            nbetas[i] = patches[i].nbeta
            ngammas[i] = patches[i].ngamma
            nks[i] = patches[i].nk

        beta1 = np.zeros(n)
        beta2 = np.zeros(n)
        sgamma1 = np.zeros(n)
        sgamma2 = np.zeros(n)
        cgamma11 = np.zeros(n)
        cgamma12 = np.zeros(n)
        cgamma21 = np.zeros(n)
        cgamma22 = np.zeros(n)
        k11 = np.zeros(n)
        k12 = np.zeros(n)
        k21 = np.zeros(n)
        k22 = np.zeros(n)
        p = np.ones(n)
        q = np.ones(n)

        for i in range(n):
            beta1[i] = patches[i].beta[0]
            beta2[i] = patches[i].beta[1]
            sgamma1[i] = patches[i].sgamma[0]
            sgamma2[i] =  patches[i].sgamma[1]
            cgamma11[i] = patches[i].cgamma[0]
            cgamma12[i] = patches[i].cgamma[1]
            cgamma21[i] = patches[i].cgamma[2]
            cgamma22[i] = patches[i].cgamma[3]
            k11[i] = patches[i].K[0]
            k12[i] = patches[i].K[1]
            k21[i] = patches[i].K[2]
            k22[i] = patches[i].K[3]
            p[i] = patches[i].p
            q[i] = patches[i].q


        self.v0 = v0s
        self.r = rs
        self.nbeta = nbetas
        self.ngamma = ngammas
        self.nk = nks
        self.beta = [beta1, beta2]
        self.sgamma = [sgamma1, sgamma2]
        self.cgamma = [cgamma11, cgamma12, cgamma21, cgamma22]
        self.K = [k11, k12, k21, k22]
        self.p = p
        self.q = q
        self.d = d
        self.M = M
        self.epsilon = patches[0].epsilon

        R0s = np.zeros(n)
        for i in range(n):
            R0s[i] = patches[i].R0
        self.R0s = R0s


        self.lambda1_2 = [patches[i].invasion_fitness()[0] for i in range(n)]
        self.lambda2_1 = [patches[i].invasion_fitness()[1] for i in range(n)] 


    def solution(self, tspan):
        return solve(system, tspan, self.v0, self.r, self.beta, self.sgamma, self.cgamma, self.K, self.p, self.q, self.d, self.M)

    def measures(self, tspan):
        return analysis(system, tspan, self.v0, self.r, self.nbeta, self.beta, self.ngamma, self.sgamma, self.cgamma, self.nk,  self.K, self.p, self.q, self.M, self.d, self.epsilon)

    def mean_replicator(self, tau, z0):
        repli, info = integrate.odeint(replicator1, z0, tau, args = (self.Theta(), self.meanInvasionFitness()[0], self.meanInvasionFitness()[1]), full_output=True)
        return repli

        return 
    def meanR0(self):
        return np.mean(self.R0s)

    def avgInvasionfitness(self):
        return np.mean(self.lambda1_2), np.mean(self.lambda2_1)

    def mean_b(self):
        b1 = self.beta[0]/(self.epsilon * self.nbeta) - 1/self.epsilon
        b2 = self.beta[1]/(self.epsilon * self.nbeta) - 1/self.epsilon
        return np.array([np.mean(self.nbeta*b1)/np.mean(self.beta), np.mean(self.nbeta*b2)/np.mean(self.beta)])

    def mean_nu(self):
        nu1 = self.sgamma[0]/(self.epsilon * self.ngamma) - 1/self.epsilon
        nu2 = self.sgamma[1]/(self.epsilon * self.ngamma) - 1/self.epsilon
        return np.array([np.mean(self.ngamma * nu1)/np.mean(self.ngamma), np.mean(self.ngamma * nu2)/np.mean(self.ngamma)])

    def mean_u(self):
        u11 = self.cgamma[0]/(self.epsilon * self.ngamma) - 1/self.epsilon
        u12 = self.cgamma[1]/(self.epsilon * self.ngamma) - 1/self.epsilon
        u21 = self.cgamma[2]/(self.epsilon * self.ngamma) - 1/self.epsilon
        u22 = self.cgamma[3]/(self.epsilon * self.ngamma) - 1/self.epsilon
        return np.array([np.mean(self.ngamma * u11)/np.mean(self.ngamma), np.mean(self.ngamma * u12)/np.mean(self.ngamma), np.mean(self.ngamma * u21)/np.mean(self.ngamma), np.mean(self.ngamma * u22)/np.mean(self.ngamma)])

    def mean_eta(self):
        eta11 = 1/(2*self.epsilon)
        eta12 = (self.p - 0.5)/self.epsilon
        eta21 = (self.q - 0.5)/self.epsilon
        eta22 = 1/(2*self.epsilon)
        return np.array([np.mean(self.beta[0]*eta11)/np.mean(self.beta[0]), np.mean(self.beta[0]*eta12)/np.mean(self.beta[0]), np.mean(self.beta[1]*eta21)/np.mean(self.beta[1]), np.mean(self.beta[1]*eta22)/np.mean(self.beta[1])])

    def mean_omega(self):
        omega11 = 1/(2*self.epsilon)
        omega12 = (self.q - 0.5)/self.epsilon
        omega21 = (self.p - 0.5)/self.epsilon
        omega22 = 1/(2*self.epsilon)
        return np.array([np.mean(self.beta[0]*omega11)/np.mean(self.beta[0]), np.mean(self.beta[0]*omega12)/np.mean(self.beta[0]), np.mean(self.beta[1]*omega21)/np.mean(self.beta[1]), np.mean(self.beta[1]*omega22)/np.mean(self.beta[1])])


    def mean_alpha(self):
        return self.epsilon**(-1)*np.array([np.mean(self.beta[0]*self.K[0])/np.mean(self.beta[0] - np.mean(self.nk)), np.mean(self.beta[1]*self.K[1])/np.mean(self.beta[1] - np.mean(self.nk)), np.mean(self.beta[0]*self.K[2])/np.mean(self.beta[0] - np.mean(self.nk)),  np.mean(self.beta[1]*self.K[3])/np.mean(self.beta[1] - np.mean(self.nk))])
        
    def meanInvasionFitness(self):
        mean_m = np.mean(self.r) + np.mean(self.ngamma)
        mean_beta = np.mean(self.nbeta)
        mean_gamma = np.mean(self.ngamma)
        mean_mu = 1/((self.meanR0() - 1)*np.mean(self.nk))
        Sstar = mean_m / mean_beta
        Tstar = 1 - Sstar
        Istar = (mean_m*Tstar)/(mean_m + mean_beta *np.mean(self.nk)*Tstar)
        Dstar = Tstar - Istar
        detP = np.linalg.det(np.array([[2*Tstar, Istar], [Dstar, Tstar]]))
        Theta1 = (2*mean_beta*Sstar*Tstar**2)/detP
        Theta2 = (mean_gamma*Istar*(Istar + Tstar))/detP
        Theta3 = (mean_gamma*Tstar*Dstar)/detP
        Theta4 = (2*mean_m*Tstar*Dstar)/detP
        Theta5 = (mean_beta*Tstar*Istar*Dstar)/detP
        Theta = Theta1 + Theta2 + Theta3 + Theta4 + Theta5
        theta1 = Theta1/Theta
        theta2 = Theta2/Theta
        theta3 = Theta3/Theta
        theta4 = Theta4/Theta
        theta5 = Theta5/Theta

        mean_lambda12 = theta1*(self.mean_b()[0] - self.mean_b()[1]) + theta2*(-self.mean_nu()[0] + self.mean_nu()[1]) + theta3*(-self.mean_u()[1] - self.mean_u()[2] + 2*self.mean_u()[3]) + theta4*(self.mean_eta()[1] - self.mean_omega()[2]) + theta5*(mean_mu*(self.mean_alpha()[2] - self.mean_alpha()[1]) + self.mean_alpha()[2] - self.mean_alpha()[3])
        mean_lambda21 = theta1*(self.mean_b()[1] - self.mean_b()[0]) + theta2*(-self.mean_nu()[1] + self.mean_nu()[0]) + theta3*(-self.mean_u()[2] - self.mean_u()[1] + 2*self.mean_u()[0]) + theta4*(self.mean_eta()[2] - self.mean_omega()[1]) + theta5*(mean_mu*(self.mean_alpha()[1] - self.mean_alpha()[2]) + self.mean_alpha()[1] - self.mean_alpha()[0])

        return [mean_lambda12, mean_lambda21]

    def Theta(self):
        mean_m = np.mean(self.r) + np.mean(self.ngamma)
        mean_beta = np.mean(self.nbeta)
        mean_gamma = np.mean(self.ngamma)
        mean_mu = 1/((self.meanR0() - 1)*np.mean(self.nk))
        Sstar = mean_m / mean_beta
        Tstar = 1 - Sstar
        Istar = (mean_m*Tstar)/(mean_m + mean_beta *np.mean(self.nk)*Tstar)
        Dstar = Tstar - Istar
        detP = np.linalg.det(np.array([[2*Tstar, Istar], [Dstar, Tstar]]))
        Theta1 = (2*mean_beta*Sstar*Tstar**2)/detP
        Theta2 = (mean_gamma*Istar*(Istar + Tstar))/detP
        Theta3 = (mean_gamma*Tstar*Dstar)/detP
        Theta4 = (2*mean_m*Tstar*Dstar)/detP
        Theta5 = (mean_beta*Tstar*Istar*Dstar)/detP
        Theta = Theta1 + Theta2 + Theta3 + Theta4 + Theta5

        return Theta