// 
// Matrix multiplication benchmark
// written by Scott B. Baden, UCSD, 4/6/11
// Modified with better command line option interface
//
// We compare the naive unblocked algorithm against the blocked version
// As an extra bonus, we also use the fast version of matrix multiplication
// found in the BLAS, e.g ATLAS, AMD ACML, Intel MKL
// (For best perforamnce in the library, we should not block, hence
// set the blocking factor = N)
// MKL will utilize all threads on the core, so this give us
// a not to exceed figure
//
// The benchmark repeats the matrix multiplication computation in order
// to help improve the accuracy of timings
// For values of N=512 or so, 5 repetitions shoudl be sufficient.
// 
// The program takes 3 command line arguments
// N, # repetitions, blocking factor
//

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <sys/time.h>
#include "types.h"
using namespace std;

double getTime();

#ifdef DEBUG
void printMatrix(Grid2D A,int n, char *msg)
#endif

#define fabs(x) ( (x)<0 ? -(x) : (x) )


void cmdLine(int argc, char *argv[], int& n, int &nreps, int &cores);


extern void mm_blas(Grid2D A, Grid2D B, Grid2D C, int n, int nreps, double& time_blas, _DOUBLE_ epsilon);


Grid2D Alloc2D(int nx,int ny, const char *mesg){

   Grid2D U = (Grid2D) malloc(sizeof(_DOUBLE_ *RESTRICT)*ny + sizeof(_DOUBLE_)*nx*ny);
   assert(U);
   if (!U)
      cerr << mesg << endl;
   else{
       int j;
       for(j=0;j<ny;j++)
           U[j] = ( _DOUBLE_  *RESTRICT  ) (U+ny) + j*nx;
   }
   return(U);
}

int main(int argc, char **argv)
{

// command line arguments:
// N, # repetitions, blocking factor
int n, nreps, cores;
cmdLine(argc, argv,  n, nreps, cores);
// mkl_set_num_threads(4);

// To avoid problems with round off, we consider two numbers
// to be equal if they differ by not more than epsilon
#ifdef _DOUBLE
_DOUBLE_ epsilon = 1.0e-8;
#else
_DOUBLE_ epsilon = 1.0e-4;
#endif


printf("\nn: %d\n",n);
printf("nreps: %d\n",nreps);

printf("\n");
// #define RESTRICT restrict
int i,j,k,r, ii, jj, kk;
Grid2D A  = Alloc2D(n,n,"A");
Grid2D B = Alloc2D(n,n,"B");
Grid2D C = Alloc2D(n,n,"C");

/* Generates a Hilbert Matrix H(i,j)
  H(I,j) = 1/(i+j+1)
  It's easy to check if the multiplication is correct;
  entry (i,j) of H * H is
  Sum(k) { 1.0/(i+k+1)*(k+j+1) }
 */

for (i=0; i<n; i++)
#pragma ivdep
    for (j=0; j<n; j++){
      B[i][j] = A[i][j] =  1.0 / (_DOUBLE_) (i+j+1);
      C[i][j] = 0.0;
    }
#ifdef DEBUG
// Print out the matrices
  char amsg[]= "--A--------";
  printMatrix("--A--------",A,n);
  char bmsg[]= "--B--------";
  printMatrix(B,n,bmsg);
#endif

_DOUBLE_ alpha = 1.0, d_one=1.0;

_DOUBLE_ sum;

// Use the blas
// Clear out C so we can use it again
double time_blas;
mm_blas(A, B, C, n, nreps, time_blas, epsilon);

printf("\n");
printf("\nTimes:\n");

time_blas /= (double) nreps;
double tg_blas = time_blas;
tg_blas /= (double) n; tg_blas /= (double) n; tg_blas /= (double) n;
printf("        BLAS: %f sec. [tg = %g]\n",time_blas, tg_blas);
double flops = 2*n; flops *= n; flops *= n;
double gflops_blas = (flops/time_blas)/1.0e9;

printf("\nGflop rate:  ");
    printf("%f\n\n",gflops_blas);
    printf("\n      N   Reps    GF     t_blas   tg_blas    cores\n");
    printf("@ %6d  %4d %f %f %f %d",n,nreps, gflops_blas, time_blas, tg_blas, cores);
printf("\n\n");
}
