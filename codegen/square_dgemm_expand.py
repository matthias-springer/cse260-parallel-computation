for i in xrange(1, 9):
	for j in xrange(1, 9):
		for o in xrange(2) :
			print("SQUARE_DGEMM_EXPAND(" + str(o) + ", " + str(i*8) + ", " + str(i*8) + ", " + str(j*8) + ")")

