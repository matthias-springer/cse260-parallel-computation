for a in xrange(8):
	print("a[ty][tx][" + str(a) + "] = A[I" + str(a) + "*N + k*by+tx];")

for b in xrange(8):
        print("b[ty][tx][" + str(b) + "] = B[J" + str(b) + "+N*(k*bx+ty)];")
	
