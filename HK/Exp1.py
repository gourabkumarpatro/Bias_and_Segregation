from HK import Member, Community
import matplotlib.pyplot as plt

if __name__ == "__main__" :
    n, A = 900, 10000
    with open("Results/Exp1/log.txt", 'a') as log :
        log.write("Gamma,TOC\n")
    for idx, Gamma in enumerate([0.0, 0.1, 0.2, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8,\
            2.0, 2.5, 3.0, 3.5, 4.0, 5.0]*5) : # Multiple trials
        community = Community(n, alpha=0.25, gamma=Gamma, activity=A)
        opinions, stamps, toc = community.simulate()
        with open("Results/Exp1/log.txt", 'a') as log :
            log.write("%f,%f\n" % (Gamma, toc))
        print("Gamma :", Gamma, " TOC :", toc)
        for i in range(n) :
            plt.plot(stamps[i], opinions[i])
        plt.savefig("Results/Exp1/Plot_%02d.png"%idx)
        plt.clf()
