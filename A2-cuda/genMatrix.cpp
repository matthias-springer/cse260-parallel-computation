/* Generates a Hilbert Matrix H(i,j)
  H(i,j) = 1/(i+j+1),   0 < i,j < n
  It's easy to check if the multiplication is correct;
  entry (i,j) of H * H is
  Sum(k) { 1.0/(i+k+1)*(k+j+1) }
 */

#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include "types.h"
using namespace std;


#define A(i,j) (a[i*n+j])

void genMatrix( _DOUBLE_ *a, unsigned int m, unsigned int n)
{
  unsigned int i, j;

  for ( i=0; i<m; i++ ) {
    for ( j=0; j<n; j++ ) {
      A( i,j ) =  1.0 / (_DOUBLE_) (i+j+1);
    }
  }
}

#define C(i,j) (c[i*n+j])

#define fabs(x) ( (x)<0 ? -(x) : (x) )

// Verify against exact answer
void verify( _DOUBLE_ *c, unsigned int m, unsigned int n, _DOUBLE_ epsilon, char *mesg)
{
  _DOUBLE_ error = 0.0;
  int ierror = 0;
  const int MAX_ERRORS = 20;

  // Assumes m=n
  _DOUBLE_ *fij = new _DOUBLE_[2*m];
  assert(fij);
  for (unsigned int i = 0; i < 2*m; i++){
     fij[i] = 1/(_DOUBLE_) (i+1);
  }
  for ( unsigned int i=0; i<m; i++ ) {
    for ( unsigned int j=0; j<n; j++ ) {
        _DOUBLE_ C_exact =  0;
        for (int k=n-1;k>=0; k--){
            C_exact +=  fij[i+k]*fij[j+k];
//            printf("i,j,k: %d, %d, %d: %f\n",i,j,k,C_exact);
        }

            _DOUBLE_ delta = fabs( C( i,j ) - C_exact);
            if ( delta > epsilon ) {
                ierror++;
	        error += delta;
                if (ierror == 1)
                    cout << "Error report for " << mesg << ":" << endl;
                if (ierror <= MAX_ERRORS)
                    cout << "C[" << i << ", " << j << "] is " << C(i,j) << ", should be: " << C_exact << endl;
            }
    }
  }

  /* 	Normalize the error */
  error /= (_DOUBLE_) (n*n);

  if ( ierror  ){
    cout << "  *** A total of " << ierror  << " differences, error = " << error;
  }
  else{
      cout << endl << mesg << ": ";
      cout << "answers matched to within " << epsilon;
  }
  cout << endl << endl;
  delete [] fij;
}  

#define C_h(i,j) (c_h[i*n+j])
#define C_d(i,j) (c_d[i*n+j])

// Verify host result against device result
void verify( _DOUBLE_ *c_d, _DOUBLE_ *c_h, unsigned int m, unsigned int n, _DOUBLE_ epsilon, char *mesg)
{
  _DOUBLE_ error = 0.0;
  int ierror = 0;
  unsigned int mn = m * n;
  for ( unsigned int ij=0; ij<mn; ij++ ) {
      _DOUBLE_ diff = fabs(c_h[ij] - c_d[ij]);
      if ( diff > epsilon ) {
          ierror++;
          error += diff;
          if (ierror == 1)
            cout << "Error report for " << mesg << ":" << endl;
          if (ierror <= 10){
            int i  = ij / n;
            int j = ij % n;
            cout << "C_d[" << i << ", " << j << "] == " << C_d(i,j);
            cout << ", C_h[" << i << ", " << j << "] == " << C_h(i,j) << endl;
          }
      }
  }

  /* 	Normalize the error */
  error /= (_DOUBLE_) (n*n);

  if ( ierror  )
    cout << "  *** A total of " << ierror  << " differences, error = " << error;
  else{
      cout << endl << mesg << ": ";
      cout << "answers matched to within " << epsilon;
  }
  cout << endl << endl;
}  
void printMatrix( _DOUBLE_ *a, unsigned int m, unsigned int n)
{
  unsigned int i, j;

  cout.precision(2);
  cout.width(5);
  for ( i=0; i<m; i++ ) {
    for ( j=0; j<n; j++ ) 
        cout << A(i,j) << " ";
//      printf("%5.2f ", A( i,j ));
    cout << endl;
//    printf("\n");
  }
}
