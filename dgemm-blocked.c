/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

#include <emmintrin.h>


const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#endif

#define min(a,b) (((a)<(b))?(a):(b))

#define do_block_not_unrolled(i, j, k, M, N, K) \
        do_block(lda, (M), (N), (K), A + (i)*lda + (k), B + (k)*lda + (j), C + (i)*lda + (j));

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i*lda+j];
      for (int k = 0; k < K; ++k)
	cij += A[i*lda+k] * B[k*lda+j];
      C[i*lda+j] = cij;
    }
}

// #define INNER_DOBLOCK_LOOP 1

#define MU 4
#define NU 2
#define KU 4

#define BUFFER_A 1

double* A_buffered;
double* B_transposed;
static void do_block_unrolled (int lda, int BLOCK_SIZE_REGISTER, double* A, double* B, double* C)
{
	// transpose matrix B
	for (int k = 0; k < BLOCK_SIZE_REGISTER; k++)
	{
		for (int j = 0; j < BLOCK_SIZE_REGISTER; j++)
		{
			B_transposed[BLOCK_SIZE_REGISTER*j+k] = B[k*lda+j];
		}
	}

#ifdef BUFFER_A
	// buffer matrix A
	for (int i = 0; i < BLOCK_SIZE_REGISTER; i++) {
		for (int k = 0; k < BLOCK_SIZE_REGISTER; k++) {
			A_buffered[i*BLOCK_SIZE_REGISTER + k] = A[i*lda + k];
		}
	}
#endif

	double c_d_1;
	double c_d_2;
	
  /* For each row i of A */
#ifdef INNER_DOBLOCK_LOOP
  for (int i = 0; i < BLOCK_SIZE_REGISTER; i += MU)
#endif
#ifndef INNER_DOBLOCK_LOOP
	for (int i = 0; i < BLOCK_SIZE_REGISTER; ++i)
#endif
    /* For each column j of B */
#ifdef INNER_DOBLOCK_LOOP
    for (int j = 0; j < BLOCK_SIZE_REGISTER; j += NU)
#endif
#ifndef INNER_DOBLOCK_LOOP
		for (int j = 0; j < BLOCK_SIZE_REGISTER; ++j)
#endif
    {
      /* Compute C(i,j) */
#ifndef INNER_DOBLOCK_LOOP
      double cij = C[i*lda+j];
#endif

#ifdef INNER_DOBLOCK_LOOP
      for (int k = 0; k < BLOCK_SIZE_REGISTER; k += KU) {
#endif
#ifndef INNER_DOBLOCK_LOOP
			for (int k = 0; k < BLOCK_SIZE_REGISTER; k += 8) {
#endif
#ifdef INNER_DOBLOCK_LOOP

				for (int k00 = k; k00 < k+KU; k00++) {
							int index_A = i*BLOCK_SIZE_REGISTER + k;
							int index_B = j*BLOCK_SIZE_REGISTER + k;

							//     cij
							double c00 = A_buffered[index_A] * B_transposed[index_B];
							double c01 = A_buffered[index_A] * B_transposed[index_B + BLOCK_SIZE_REGISTER];
							double c10 = A_buffered[index_A + BLOCK_SIZE_REGISTER] * B_transposed[index_B];
							double c11 = A_buffered[index_A + BLOCK_SIZE_REGISTER] * B_transposed[index_B + BLOCK_SIZE_REGISTER];
							double c20 = A_buffered[index_A + 2 * BLOCK_SIZE_REGISTER] * B_transposed[index_B];
							double c21 = A_buffered[index_A + 2 * BLOCK_SIZE_REGISTER] * B_transposed[index_B + BLOCK_SIZE_REGISTER];
							double c30 = A_buffered[index_A + 3 * BLOCK_SIZE_REGISTER] * B_transposed[index_B];
							double c31 = A_buffered[index_A + 3 * BLOCK_SIZE_REGISTER] * B_transposed[index_B + BLOCK_SIZE_REGISTER];

							C[i*lda + j] += c00;
							C[i*lda + j+1] += c01;
							C[(i+1)*lda + j] += c10;
							C[(i+1)*lda + j+1] += c11;
							C[(i+2)*lda + j] += c20;
							C[(i+2)*lda + j+1] += c21;
							C[(i+3)*lda + j] += c30;
							C[(i+3)*lda + j+1] += c31;

//					for (int j00 = j; j00 < j + NU; j00++) {			// 2
//						for (int i00 = i; i00 < i + MU; i00++) {		// 4
//							C[i00*lda + j00] += A[i00*lda + k00] * B_transposed[j00*BLOCK_SIZE_REGISTER + k00];
//						}
//					}
				}
#endif

#ifndef INNER_DOBLOCK_LOOP
				int index_A = i*BLOCK_SIZE_REGISTER + k;
				int index_B = j*BLOCK_SIZE_REGISTER + k;
			
				__m128d a1 = _mm_load_pd(A_buffered + index_A);
				__m128d a2 = _mm_load_pd(A_buffered + index_A + 2);
        __m128d a3 = _mm_load_pd(A_buffered + index_A + 4);
        __m128d a4 = _mm_load_pd(A_buffered + index_A + 6);

        __m128d b1 = _mm_load_pd(B_transposed + index_B);
        __m128d b2 = _mm_load_pd(B_transposed + index_B + 2);
        __m128d b3 = _mm_load_pd(B_transposed + index_B + 4);
        __m128d b4 = _mm_load_pd(B_transposed + index_B + 6);

				__m128d d1 = _mm_mul_pd(a1, b1);
				__m128d d2 = _mm_mul_pd(a2, b2);
        __m128d d3 = _mm_mul_pd(a3, b3);
        __m128d d4 = _mm_mul_pd(a4, b4);

				__m128d c1 = _mm_add_pd(d1, d2);
				__m128d c2 = _mm_add_pd(d3, d4);
				__m128d c3 = _mm_add_pd(c1, c2);
				
/*
				double cij1 = A[index_A] * B_transposed[index_B];
			  double cij2 = A[index_A + 1] * B_transposed[index_B + 1];
				double cij3 = A[index_A + 2] * B_transposed[index_B + 2];
				double cij4 = A[index_A + 3] * B_transposed[index_B + 3];
				//double cres = 

			  cij += cij1 + cij2 + cij3 + cij4;	 //+ cij5 + cij6; // cij6 + cij7 + cij8;
//			  cij += c_d_1 + c_d_2;
*/

					cij += _mm_cvtsd_f64(c3);
					__m128d c4 = _mm_unpackhi_pd(c3, c1);
					cij += _mm_cvtsd_f64(c4);

#endif

			}

#ifndef INNER_DOBLOCK_LOOP
      C[i*lda+j] = cij;
#endif

    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
	// will always be BLOCK_SIZE, but the compiler is weired
	register int BLOCK_SIZE_REGISTER = min(lda * 100000, BLOCK_SIZE);
	int fringe_start = lda / BLOCK_SIZE_REGISTER * BLOCK_SIZE_REGISTER;

	B_transposed = (double*) memalign(16, sizeof(double) * BLOCK_SIZE_REGISTER*BLOCK_SIZE_REGISTER);
	A_buffered = (double*) memalign(16, sizeof(double) * BLOCK_SIZE_REGISTER*BLOCK_SIZE_REGISTER);

  /* For each block-row of A */ 
  for (int i = 0; i < fringe_start; i += BLOCK_SIZE_REGISTER)
    /* For each block-column of B */
    for (int j = 0; j < fringe_start; j += BLOCK_SIZE_REGISTER)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < fringe_start; k += BLOCK_SIZE_REGISTER)
      {
				do_block_unrolled(lda, BLOCK_SIZE_REGISTER, A + i*lda + k, B + k*lda + j, C + i*lda + j);
				//do_block_not_unrolled(i, j, k, BLOCK_SIZE_REGISTER, BLOCK_SIZE_REGISTER, BLOCK_SIZE_REGISTER);
				//do_block_not_unrolled(i, j, k, min(lda-i, BLOCK_SIZE_REGISTER), min(lda-j, BLOCK_SIZE_REGISTER), min(lda-k, BLOCK_SIZE_REGISTER));

      }

	/* For the fringe cases */
	// {i}
	{
		int i = fringe_start;
		for (int j = 0; j < fringe_start; j += BLOCK_SIZE_REGISTER)
			for (int k = 0; k < fringe_start; k += BLOCK_SIZE_REGISTER) 
			{
				do_block_not_unrolled(i, j, k, lda - i, BLOCK_SIZE_REGISTER, BLOCK_SIZE_REGISTER);
			}
	}

	// {j}
  {
    int j = fringe_start;
    for (int i = 0; i < fringe_start; i += BLOCK_SIZE_REGISTER)
      for (int k = 0; k < fringe_start; k += BLOCK_SIZE_REGISTER)
      {
				do_block_not_unrolled(i, j, k, BLOCK_SIZE_REGISTER, lda - j, BLOCK_SIZE_REGISTER);
      }
  }

	// {k}
  {
    int k = fringe_start;
    for (int i = 0; i < fringe_start; i += BLOCK_SIZE_REGISTER)
      for (int j = 0; j < fringe_start; j += BLOCK_SIZE_REGISTER)
      {
				do_block_not_unrolled(i, j, k, BLOCK_SIZE_REGISTER, BLOCK_SIZE_REGISTER, lda - k);
      }
  }

	// {i, j}
	{
		int i = fringe_start;
		int j = fringe_start;
		for (int k = 0; k < fringe_start; k += BLOCK_SIZE_REGISTER) 
		{
			do_block_not_unrolled(i, j, k, lda - i, lda - j, BLOCK_SIZE_REGISTER);
		}
	}

	// {j, k}
  {
    int j = fringe_start;
    int k = fringe_start;
    for (int i = 0; i < fringe_start; i += BLOCK_SIZE_REGISTER) 
    {
      do_block_not_unrolled(i, j, k, BLOCK_SIZE_REGISTER, lda - j, lda - k);
    }
  }

	// {i, k}
  {
    int i = fringe_start;
    int k = fringe_start;
    for (int j = 0; j < fringe_start; j += BLOCK_SIZE_REGISTER)
    {
      do_block_not_unrolled(i, j, k, lda - i, BLOCK_SIZE_REGISTER, lda - k);
    }
  }

	// {i, j, k}
	{
		int i = fringe_start;
		int j = fringe_start;
		int k = fringe_start;

		do_block_not_unrolled(i, j, k, lda - i, lda - j, lda - k);
	}

}
