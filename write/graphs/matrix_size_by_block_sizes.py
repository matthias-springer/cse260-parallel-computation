import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from os import listdir

l1 = []
l2 = []

l3 = []
l4 = []

ref1 = []
ref2 = []

#files = [ f for f in listdir('profiles/') if (f.find('.pdf') == -1 and f.find('BLAS') == -1)]
files = ['PROFILE_UNROLL_j_2_k_4', 'PROFILE_UNROLL_k_4']
print files
counter = 1

with open("profiles/PROFILE_BLOCKED") as f:
	if True:
		content = f.readlines()
		
		first = True
		
		for line in content:
			if first:
				first = False
			else:
				vals = line.split()
				ref1 += [int(vals[0])]
				ref2 += [float(vals[1])]
				
			counter += 1
	
			if counter == 800:
				break

with open("profiles/PROFILE_UNROLL_j_2_k_4") as f:
	if True:
		content = f.readlines()
		
		first = True
		
		for line in content:
			if first:
				first = False
			else:
				vals = line.split()
				l3 += [int(vals[0])]
				l4 += [float(vals[1])]
				
			counter += 1
	
			if counter == 800:
				break
				
with open("profiles/PROFILE_UNROLL_k_4") as f:
	if True:
		content = f.readlines()
		
		first = True
		
		for line in content:
			if first:
				first = False
			else:
				vals = line.split()
				l1 += [int(vals[0])]
				l2 += [float(vals[1])]
				
			counter += 1
	
			if counter == 800:
				break
				
if True:		
	#plt.clf()
	plt.figure(figsize=(1000, 1000))
	p1, = plt.plot(l1, l2)
	p2, = plt.plot(l3, l4)
	p3, = plt.plot(ref1, ref2)
	#plt.legend( [p1,p2],['dgemm-blocked (' + paramij + ', ' + paramij + ', ' + paramk + ')' , 'dgemm-blas'], 4)
	plt.legend( [p1,p2, p3],['dgemm-blocked (parameter-tuned, k += 4)', 'dgemm-blocked (parameter-tuned, j += 2, k += 4)' , 'dgemm-blocked (parameter-tuned, i += 2, j += 2, k += 4)'], 4)

	plt.ylabel('GFlops')
	plt.xlabel('matrix size (n)')
	plt.xlim([0, 800])
	#pp = PdfPages('profiles/PROFILE_OUTUT_' + paramij + '_' + paramk + '.pdf')
	pp = PdfPages('profiles/PROFILE_UNROLLING_SSE.pdf')
	plt.savefig(pp, format='pdf')
	pp.close()

	plt.show()