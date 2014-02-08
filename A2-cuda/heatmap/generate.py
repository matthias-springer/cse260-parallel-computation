import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import ImageGrid
import math

f = open("results.txt",'r')
lines = f.readlines() 

results = [[[0 for x in xrange(11)] for x in xrange(11)] for x in xrange(4)]

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
		
	results[bucket][int(math.log(x, 2))][int(math.log(y, 2))] = gf
	
# Generate some test data

F = plt.figure(1, (10, 10))
F.clf()
ZS = [1,2,3,4]

grid = ImageGrid(F, 212,
                      nrows_ncols = (1, 4),
                      direction="row",
                      axes_pad = 0.05,
                      add_all=True,
                      label_mode = "L",
                      share_all = True,
                      cbar_location="right",
                      cbar_mode="single",
                      cbar_size="10%",
                      cbar_pad=0.05,
                      )

heatmap, xedges, yedges = results[0], [1,2],[3,4]
extent = [1,11,1,11]

norm = matplotlib.colors.Normalize(vmax=100, vmin=0)
im = 0

ctr = 0

for n in [0,1,2,3]:
    size = 2**(n+8)
    for x in xrange(11):
        for y in xrange(11):
            print str(size) + "," + str(2**x) + "," + str(2**y) + "," + str(results[n][x][y])

for ax, z in zip(grid, ZS):
    data = np.ma.masked_less_equal(np.array(results[ctr]), 0)
    im = ax.pcolor(data, edgecolors='white', linewidths=1 )
    ax.grid(True, which='minor')
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)
#    ax.cax.set_xticks([1,10,20,25,30])
    ax.set_xticklabels(["1","2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"], minor=False,rotation=90)
    ax.set_xticks(np.arange(12)+0.5, minor=False)
    ax.set_xticks(np.arange(12), minor=True)
    ax.set_yticks(np.arange(12), minor=True)
    ax.set_yticklabels(["1","2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"], minor=False)
    ax.set_yticks(np.arange(12)+0.5, minor=False)
    ax.set_ylim([0, 11])
    ax.set_xlim([0, 11])
    #ax.patch.set_hatch('x')
    #ax.xticks(rotation=70)
    ctr += 1

plt.savefig("original_heat.pdf")
plt.show()
