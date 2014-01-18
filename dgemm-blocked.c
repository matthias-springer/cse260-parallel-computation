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
	
  /* For each row i of A */
  for (int i = 0; i < BLOCK_SIZE_REGISTER; ++i)
    /* For each column j of B */
    for (int j = 0; j < BLOCK_SIZE_REGISTER; ++j)
    {
      /* Compute C(i,j) */
      double cij = C[i*lda+j];
      for (int k = 0; k < BLOCK_SIZE_REGISTER; k += 4) {
				int index_A = i*lda + k;
				int index_B = j*BLOCK_SIZE_REGISTER + k;
			
				__m128d a1 = _mm_loadu_pd(A + index_A);
				__m128d a2 = _mm_loadu_pd(A + index_A + 2);
        __m128d b1 = _mm_loadu_pd(B + index_B);
        __m128d b2 = _mm_loadu_pd(B + index_B + 2);
				
				//__m128d d1 = _mm_mul_pd(a1, b1);
				//__m128d d2 = _mm_mul_pd(a2, b2);

				__m128d c = _mm_add_pd(_mm_mul_pd(a1, b1), _mm_mul_pd(a2, b2));

				double c1;
				double c2;

				_mm_storeh_pd(&c1, c);
				_mm_storel_pd(&c2, c);
				
				//double cij1 = A[index_A] * B_transposed[index_B];
				//double cij2 = A[index_A + 1] * B_transposed[index_B + 1];
				//double cij3 = A[index_A + 2] * B_transposed[index_B + 2];
				//double cij4 = A[index_A + 3] * B_transposed[index_B + 3];
				//double cres = 
			  //cij += cij1 + cij2 + cij3 + cij4;	 //+ cij5 + cij6; // cij6 + cij7 + cij8;
			  cij += c1 + c2;
			}

      C[i*lda+j] = cij;
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
