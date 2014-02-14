#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "types.h"
using namespace std;

double getTime();
extern void verify(Grid2D A, int n, char *Name, char *msg, _DOUBLE_ epsilon, int nreps);

void mm_blas(Grid2D A, Grid2D B, Grid2D C, int n, int nreps, double& time_blas, _DOUBLE_ epsilon)
{
#ifndef NO_BLAS
  int i, j, r, ii, jj, kk;
    for (i=0; i<n; i++)
#pragma ivdep
      for (j=0; j<n; j++)
	C[i][j] = 0.0;

    time_blas = -getTime(); // Start the timer
    for (r=0; r<nreps; r++){
        ii=jj=kk=0;
	    /*
	     * Users of the CBLAS interface: be aware that the CBLAS are just a C
	     * interface to the BLAS, which is based on the FORTRAN standard and
	     * subject to the FORTRAN standard restrictions. In particular, the output
	     * parameters should not be referenced through more than one argument
	     * http://software.intel.com/sites/products/documentation/hpc/compilerpro/en-us/cpp/lin/mkl/refman/appendices/mkl_appD_Intro.html
	     */

        /* Set up the call for dgemm, to perform the  matrix  multiplication
        *
        * You should not change the definitions for
        *         Alpha, and Beta 
        *         LDA, LDB, LDC
        *         transA, transB
        *
        * You need not change M, N, and K, unless your local sub-matrices are
        * not square
        */
	    const _DOUBLE_ Alpha = 1.0;
	    const _DOUBLE_ Beta  = 1.0;
	    const int M = n, N=n, K=n;
	    const int LDA = n, LDB = n, LDC = n;
	    const CBLAS_TRANSPOSE transA = CblasNoTrans;
	    const CBLAS_TRANSPOSE transB = CblasNoTrans;
	    /* Don't change this call */
	    _DOUBLE_ *RESTRICT _A = &A[ii][kk];
	    _DOUBLE_ *RESTRICT _B = &B[kk][jj];
	    _DOUBLE_ *RESTRICT _C = &C[ii][jj];
#ifdef _DOUBLE
	    cblas_dgemm( CblasRowMajor, transA, transB, M, N, K,
#else
	    cblas_sgemm( CblasRowMajor, transA, transB, M, N, K,
#endif
			 Alpha, _A, LDA,
			 _B, LDB, 
			 Beta, _C, LDC);
	    
	  }
    time_blas += getTime(); // Stop the timer
    // Verify the results
    char c_msg[] = "C";
    char msg[] = "Matrix multiplication using BLAS";
    verify(C,n,c_msg,msg,epsilon,nreps);
#ifdef DEBUG
    printMatrix(C,n,"--C--------");
#endif
#endif
}
