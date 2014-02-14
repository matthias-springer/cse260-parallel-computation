import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import ImageGrid
import math

f = open("results_multi.txt",'r')
lines = f.readlines() 

results = [[0 for x in xrange(32)] for x in xrange(8)]

for line in lines:
	x = int(line.split()[6])
	gf = float(line.split()[3])
	
	bucket = (int(line.split()[1]) - 512)/32

	if x > 0:
		print bucket, x-1	
		results[x-1][bucket] = gf

for n in range(8):
    for x in range(32):
        print(str(str((x)*32 + 512)) + "," + str(n + 1) + "," + str(results[n][x]))
	
# Generate some test data

fig, ax = plt.subplots()
heatmap = ax.pcolor(np.ma.masked_less_equal(np.array(results), 0), edgecolors='white', linewidths=1)
ax.grid(True, which='minor')
ax.set_xticklabels([str((x)*32 + 512) for x in xrange(32)], minor=False,rotation=90)
ax.set_xticks(np.arange(32)+0.5, minor=False)
ax.set_xticks(np.arange(32), minor=True)
ax.set_yticks(np.arange(8), minor=True)
ax.set_yticklabels(["1","2", "3", "4", "5", "6", "7", "8"], minor=False)
ax.set_yticks(np.arange(8)+0.5, minor=False)
#ax.set_ylim([0, 11])
ax.set_xlim([0, 32])

cb = plt.colorbar(heatmap)

"""
for y in range(np.array(results).shape[0]):
    for x in range(np.array(results).shape[1] - 1):
        plt.text(x + 0.5, y + 0.5, '%.4f' % results[y][x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )
"""

plt.savefig("multi_heat.pdf")
plt.show()

