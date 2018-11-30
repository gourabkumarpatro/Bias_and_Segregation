import numpy as np
import time
import math
import random
import operator
import matplotlib.pyplot as plt
def get_col(arr,col):
        return [row[col] for row in arr]
N = 100
R = 10
np.random.seed(int(time.time()))
random.seed(int(time.time()))
Matrix = np.random.rand(N,N)
Matrix = Matrix/Matrix.sum(axis=1)[:,None]
T = []
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
print(len(T))
lam = random.uniform(0,1)
Opinion = [[0 for x in range(R)] for y in range(N)]
Opinion0 = [[0 for x in range(R)] for y in range(N)]
for it1 in range(N):
        for it2 in range(R):
                temp = random.uniform(0,1)
                Opinion[it1][it2] = temp
                Opinion0[it1][it2] = temp
C = [[0 for x in range(R)] for y in range(R)]
lam2 = random.uniform(0.7,0.9)
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
gamma = 2
for tim,it in T: 
        if count%500 == 0:
                print(count)
                print(Opinion)
                if count%2000 == 0:
                        formatted = (np.array(Opinion) * 255).astype('uint8')
                        for it in range(0,R):
                                fig, ax = plt.subplots()
                                im = ax.imshow([get_col(formatted,it)])
                                ax.set_aspect('auto')
                                fig.colorbar(im,ax=ax)
                                fig.savefig('./Images3/step_'+str(count)+' '+str(it)+'.png')
                                fig.clf()
                                plt.close(fig)
        count = count + 1
        sum1 = 0
        sum2 = 0
        prev = Opinion[it]
        for it1 in range(N):
                sum3 = 0
                for it2 in range(R):
                        sum3 += pow(Opinion[it][it2]-Opinion[it1][it2],2)
                sum3 = math.sqrt(sum3)
                sum2 += pow(Matrix[it][it1]/(sum3+0.00001),gamma)
        for it1 in range(N):
                sum3 = 0
                for it2 in range(R):
                        sum3 += pow(Opinion[it][it2]-Opinion[it1][it2],2)
                sum3 = math.sqrt(sum3)                
                sum1 += pow(Matrix[it][it1]/(sum3+0.000001),gamma)*np.matrix(Opinion[it1])
        temp1 = np.matrix(sum1)
        temp2 = lam*np.matmul(np.matrix(C),temp1.transpose())
        temp3 = (1-lam)*np.matrix(Opinion0[it])
        temp = temp2.transpose()+temp3
        Opinion[it] = temp.tolist()[0]
        f.write(str(tim)+" "+str(it)+" \n")
f.close()
