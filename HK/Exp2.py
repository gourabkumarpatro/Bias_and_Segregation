
from HK import Member, Community
import matplotlib.pyplot as plt
import random

if __name__ == "__main__" :
    n, A = 900, 10000
    for idx, Gamma in enumerate([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,\
        0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] * 2) : # Multiple trials
        community = Community(n, alpha=0.25, gamma=Gamma, activity=A)
        for member in community.members :
            member.opinion += random.randrange(2) - 0.5
            member.epsilon *= 2
        opinions, stamps, _ = community.simulate()
        with open("Results/Exp2/log.txt", 'a') as log :
            log.write("%d,%f\n" % (idx, Gamma))
        print("\nGamma :", Gamma)
        for i in range(n) :
            plt.plot(stamps[i], opinions[i])
        plt.savefig("Results/Exp2/Plot_%02d.png"%idx)
        plt.clf()
