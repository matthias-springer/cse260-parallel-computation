#ifndef types_h
/* Do not change the code in this file, as doing so
 * could cause your submission to be graded incorrectly
 */

/*
 * Include this in every module that uses floating point
 * arithmetic, and declare all floating point values as "_DOUBLE_"
 * With a switch of a command line macro set up in the Makefile
 * we can then change the arithmetic
*/
#define types_h
#ifndef _DOUBLE
#define _DOUBLE_ float
#else
#define _DOUBLE_ double
#endif
#else
#endif

#ifdef __RESTRICT
// Intel uses a different syntax than the Gnu compiler
#if __ICC
#define RESTRICT restrict
#else
// Gnu uses a different syntax than the Intel compiler
#define RESTRICT __restrict
#endif
#else
// Turns the restrict keyword into the empty string if not selected
#define RESTRICT
#endif
typedef _DOUBLE_ *RESTRICT *RESTRICT Grid2D;
typedef _DOUBLE_ *RESTRICT *RESTRICT *RESTRICT Grid3D;

#ifdef USE_MKL
#include "mkl.h"
#endif

#ifdef __INTEL_COMPILER
#ifndef NO_BLAS
#include "mkl_cblas.h"
#endif
#else

#ifndef NO_BLAS
extern "C" 
{
#include "cblas.h"
}
#endif
#endif

#include <stdint.h>
