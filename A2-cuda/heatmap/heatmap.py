import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import ImageGrid
import math

f = open("results.txt",'r')
lines = f.readlines() 

results = [[0 for x in xrange(6)] for x in xrange(4)]

for line in lines:
	x = float(line.split()[2])
	y = float(line.split()[3])
	gf = float(line.split()[8])
	
	bucket = 100
	if int(line.split()[1]) == 256:
		bucket = 0
	elif int(line.split()[1]) == 512:
		bucket = 1
	elif int(line.split()[1]) == 1024:
		bucket = 2
	elif int(line.split()[1]) == 2048:
		bucket = 3
	
	print bucket
	print int(math.log(x, 2))

	results[bucket][int(math.log(x, 2))] = gf

for n in range(4):
    for x in range(6):
        print(str(2**(8+n)) + "," + str(2**x) + "," + str(results[n][x]))
	
# Generate some test data

fig, ax = plt.subplots()
heatmap = ax.pcolor(np.ma.masked_less_equal(np.array(results), 0), edgecolors='white', linewidths=1)
ax.grid(True, which='minor')
ax.set_xticklabels(["1","2", "4", "8", "16", "32"], minor=False,rotation=0)
ax.set_xticks(np.arange(6)+0.5, minor=False)
ax.set_xticks(np.arange(6), minor=True)
ax.set_yticks(np.arange(4), minor=True)
ax.set_yticklabels(["256","512", "1024", "2048"], minor=False)
ax.set_yticks(np.arange(4)+0.5, minor=False)
#ax.set_ylim([0, 11])
#ax.set_xlim([0, 11])

cb = plt.colorbar(heatmap)

for y in range(np.array(results).shape[0]):
    for x in range(np.array(results).shape[1] - 1):
        plt.text(x + 0.5, y + 0.5, '%.4f' % results[y][x],
                 horizontalalignment='center',
                 verticalalignment='center',
                 )

plt.savefig("blocked_heat.pdf")
plt.show()

