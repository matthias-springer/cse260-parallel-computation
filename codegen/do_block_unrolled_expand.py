for i in xrange(1, 9):
	for j in xrange(1, 9):
		for o in xrange(2) :
			print("DO_BLOCK_UNROLLED_EXPAND(" + str(o) + ", " + str(i*8) + ", " + str(i*8) + ", " + str(j*8) + ")")

