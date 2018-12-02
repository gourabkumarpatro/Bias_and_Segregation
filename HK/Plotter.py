import numpy as np
import matplotlib.pyplot as plt

times = {}
with open('Results/Exp1/log.txt', 'r') as f :
    for line_no, line in enumerate(f) :
        if line_no == 0 : continue
        x = line[:-1].split(',')
        gamma, toc = float(x[0]), float(x[1])
        if gamma not in times : times[gamma] = [toc]
        else : times[gamma] += [toc]

# for v in times.values() : v.sort()
def aggregate(tocs) :
    return sum(tocs)/len(tocs) # Average
    # return tocs[len(tocs)//2] # Median

plots = [ (gamma, aggregate(tocs)) for (gamma, tocs) in times.items()]
plots.sort(key=(lambda tup: tup[0]))

plt.plot([x for x,_ in plots], [y for _,y in plots], 'r')
plt.plot([x for x,_ in plots], [y for _,y in plots], 'ro')
plt.savefig('Results/Exp1/trend.png')
