import numpy as np
import time
import math
import random
import operator
import matplotlib.pyplot as plt
def get_col(arr,col):
        return [row[col] for row in arr]
N = 900  # number of actors
R = 10   # number of issues
np.random.seed(int(time.time()))
random.seed(int(time.time()))
# Initialization for experiment 1
Matrix = np.random.rand(N,N)           # row stochastic matrix depicting graph
Matrix = Matrix/Matrix.sum(axis=1)[:,None]
# Initialization for experiment 2
#Matrix = np.ones((N,N))/math.sqrt(N)
print(Matrix)
T = []
# Generation of interaction times of all actors
for it in range(0,N):
        start = 0
        count = 0
        lamTemp = 1.0*(np.random.randint(0,5)+1)
        lamTempInv = 1.0/lamTemp
        arr = np.random.exponential(lamTempInv,10000)
        for it1 in arr:
                start = start + it1
                T.append((start,it))
T.sort(key=operator.itemgetter(0))
lam = random.uniform(0.7,0.9)
print(lam)
Opinion = [[0 for x in range(R)] for y in range(N)]      # opinion matrix
Opinion0 = [[0 for x in range(R)] for y in range(N)]     # initial opinion matrix at time 0
mu1, sigma1 = 0, 0.005
mu2, sigma2 = 10, 0.005
for it1 in range(R):
        # Initialization for experiment 1
        s = np.random.normal(mu1,sigma1,N)
        t = np.random.normal(mu2,sigma2,N)
        s = s + t
        mini = min(s)
        s1 = s
        if mini < 0:
                s1 = [x-mini for x in s]
        maxi = max(s1)
        s1 = [x*1.0/maxi for x in s1]
        count = 0
        
        for it2 in s1:
                Opinion[count][it1] = it2
                Opinion0[count][it1] = it2
                count += 1
        """
        Initialization for experiment 2
        for it2 in range(N):
                it3 = int(it2)/30
                it4 = int(it2)%30
                if abs(7-it3)+abs(7-it4) < 8:
                        Opinion[it2][it1] = 1.0
                        Opinion0[it2][it1] = 1.0
                elif abs(21-it3)+abs(21-it4) < 8:
                        Opinion[it2][it1] = 0.9
                        Opinion0[it2][it1] = 0.9
                else:
                        temp = np.random.uniform(0,1)
                        Opinion[it2][it1] = 0.0
                        Opinion0[it2][it1] = 0.0
        """
C = [[0 for x in range(R)] for y in range(R)]            # MiDS matrix
lam2 = random.uniform(0,1)          # linear interpolation parameter
arr = [x*(1-lam2) for x in np.random.dirichlet(np.ones(R*R-R),size=1)[0]]
kk = 0
for it1 in range(R):
        for it2 in range(R):
                if it1 == it2:
                        C[it1][it2] = lam2
                else:
                        C[it1][it2] = arr[kk]
                        kk += 1
f = open('ndimFJ1.txt','w')
count = 0
gamma = 10                                             # bias factor
for tim,it in T:                                        # tim is time, it is user
        if count%500 == 0:                              # checkpoints
                if count%2000 == 0:
                        formatted = (np.array(Opinion) * 255).astype('uint8')
                        for it in range(0,R):
                                fig, ax = plt.subplots()
                                im = ax.imshow(np.resize([get_col(formatted,it)],(30,30)))
                                ax.set_aspect('auto')
                                fig.colorbar(im,ax=ax)
                                fig.savefig('../g0/step_'+str(count)+' '+str(it)+'.png')
                                fig.clf()
                                plt.close(fig)
        count = count + 1
        sum1 = 0
        sum2 = 0
        prev = np.copy(np.array(Opinion))       
        for it1 in range(N):                    # get denominator of the term
                sum3 = 0
                for it2 in range(R):
                        sum3 += pow(Opinion[it][it2]-Opinion[it1][it2],2)
                sum3 = math.sqrt(sum3)
                sum2 += pow(Matrix[it][it1]/(sum3+0.000000001),gamma)
        for it1 in range(N):    
                sum3 = 0
                for it2 in range(R):
                        sum3 += pow(Opinion[it][it2]-Opinion[it1][it2],2)
                sum3 = math.sqrt(sum3)
                #Comment previous line and uncomment next line for original FJ model
                #sum1 += (pow(Matrix[it][it1]/(sum3+0.000000001),gamma)*np.matrix(Opinion[it1]))/sum2         # get modified wij*xj value and sum over j
                sum1 += Matrix[it][it1]*np.matrix(Opinion[it1])
        temp1 = np.matrix(sum1)
        temp2 = lam*np.matmul(np.matrix(C),temp1.transpose())
        temp3 = (1-lam)*np.matrix(Opinion0[it])
        temp = temp2.transpose()+temp3                 # lam*C*sig(wij(modified)*xj) + (1-lam)*x0
        Opinion[it] = temp.tolist()[0]
        diff = np.absolute(np.array(Opinion)-prev)
        check = np.sum(diff)
        if (count-1)%500 == 0:
                print(count-1,check)
        f.write(str(tim)+" "+str(it)+" \n")
f.close()
