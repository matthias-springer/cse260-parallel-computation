#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <math.h>
#include "types.h"
// Verify two arrays
// Uses external array to speed up the comparison
void verify(Grid2D A, int n, char *Name, char *msg, _DOUBLE_ epsilon, int nreps){
  int nerror=0;
  int i, j, k;
  _DOUBLE_ error = 0.0;
  const int MAX_ERRORS = 20;

  _DOUBLE_ *fij = new _DOUBLE_[2*n];
  assert(fij);
  for (i = 0; i < 2*n; i++){
     fij[i] = 1/(_DOUBLE_) (i+1);
  }
  printf("Verifying %s:\n",msg);

  _DOUBLE_ mult = (_DOUBLE_) nreps;
  for ( j=0; j<n; j++ ) {
    for ( i=0; i<n; i++ ) {
      _DOUBLE_ A_exact =  0.0;
  // Exact value is Sum(k) { 1.0/(i+k+1)*(k+j+1) }
//	A_exact += 1.0/((_DOUBLE_) (i+k+1)*(k+j+1));
      for (k=n-1;k>=0; k--){
//      for (k=0;k<n; k++){
        A_exact +=  fij[i+k]*fij[j+k];
	}
      _DOUBLE_ delta = fabs( A[i][j] - (mult * A_exact) );
      if ( delta > epsilon ) {
        nerror++;
	if (delta > error)
	    error = delta;
	// For error normalization
//	error += delta;
	if (nerror <= MAX_ERRORS){
	    fprintf(stderr,"%s[%d,%d] incorrect; sb: %21.14g, is: %21.14g [diff: %17.10g]\n",Name, i, j, mult * A_exact, A[i][j],delta);
         }
      }
    }
  }

  /* 	Normalize the error */
//  error /= (_DOUBLE_) (n*n);

  if (nerror ){
     fprintf(stderr,"\n *** # incorrect points = %d; error = %e\n", nerror, error);
     fprintf(stderr,"\n *** Epsilon = %17.5g\n",epsilon);
    }
  else
     printf("  Answers matched exactly\n\n");

  delete [] fij;
}
