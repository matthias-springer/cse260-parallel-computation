import numpy as np
import matplotlib.pyplot as plt

N = 4

# CPU, unoptimized, shared, shared+coalesced, sared+coalesced+unrolled

perfCPU = (0.800001, 0.736525, 0.408128, 0.151970)
perfUnopt = (58.3, 54.3, 56.6, 56.6)
perfSh = (53.4, 40.9, 60, 59.3)
perfShCoal = (88.8, 93.8, 94.6, 96.1)
perfShCoalUnr = (118, 127.6, 130.1, 133)

menMeans = (20, 35, 30, 35, 27)
menStd =   (2, 3, 4, 1, 2)

ind = np.arange(N)  # the x locations for the groups
width = 0.17       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, perfCPU, width, color='c')
rects2 = ax.bar(ind+width, perfUnopt, width, color='g')
rects3 = ax.bar(ind+2*width, perfSh, width, color='b')
rects4 = ax.bar(ind+3*width, perfShCoal, width, color='r')
rects5 = ax.bar(ind+4*width, perfShCoalUnr, width, color='y')

# add some
ax.set_ylabel('GFlops')
#ax.set_title('Scores by group and gender')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('n = 256', 'n = 512', 'n = 1024', 'n = 2048') )
ax.yaxis.grid(True)

ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]), ('Host CPU', 'CUDA: Unoptimized', 'CUDA: Shared Memory, Uncoalesced', 'CUDA: Shared Memory, Coalesced', 'CUDA: Shared Memory, Coalesced, Unrolled'), loc=4 )
ax.axhline(y=71.55, xmin=0, xmax=100, linewidth=3, c="gray")

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., height*1.01+1, '%4.2f'%(height),
                ha='center', va='bottom', rotation=90)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
plt.ylim((0,155))
plt.show()

