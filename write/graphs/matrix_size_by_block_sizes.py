import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from os import listdir

l1 = []
l2 = []

ref1 = []
ref2 = []

#files = [ f for f in listdir('profiles/') if (f.find('.pdf') == -1 and f.find('BLAS') == -1)]
files = ['PROFILE_BLOCKED']
print files
counter = 1

with open("profiles/PROFILE_BLAS") as f:
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



for fn in files:
	l1 = []
	l2 = []

	file = "profiles/" + fn
	paramk = file.split("_")[-1]
	paramij = file.split("_")[-2]
	print file
	
	with open(file) as f:
		content = f.readlines()
		counter = 1
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
				
	plt.clf()
	plt.figure(figsize=(1000, 1000))
	p1, = plt.plot(l1, l2)
	p2, = plt.plot(ref1, ref2)
	#plt.legend( [p1,p2],['dgemm-blocked (' + paramij + ', ' + paramij + ', ' + paramk + ')' , 'dgemm-blas'], 4)
	plt.legend( [p1,p2],['dgemm-blocked (parameter-tuned)' , 'dgemm-blas'], 4)

	plt.ylabel('GFlops')
	plt.xlabel('matrix size (n)')
	plt.xlim([0, 800])
	#pp = PdfPages('profiles/PROFILE_OUTUT_' + paramij + '_' + paramk + '.pdf')
	pp = PdfPages('profiles/PROFILE_BLOCKED.pdf')
	plt.savefig(pp, format='pdf')
	pp.close()

	plt.show()