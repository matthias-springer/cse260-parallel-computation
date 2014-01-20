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
//#define BLOCK_SIZE 32
#endif

int BLOCK_SIZE = 32;

int BLOCK_SIZE_I = 16;
int BLOCK_SIZE_J = 32;
int BLOCK_SIZE_K = 32;

#define min(a,b) (((a)<(b))?(a):(b))

#define do_block_not_unrolled(i, j, k, M, N, K) \
        do_block(lda, (M), (N), (K), A + (i)*lda + (k), B + (k)*lda + (j), C + (i)*lda + (j));

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C)
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

//double* restrict A_buffered;
//double* restrict B_transposed;

double A_buffered[32*32*sizeof(double)] __attribute__((aligned(16)));
double B_transposed[32*32*sizeof(double)] __attribute__((aligned(16)));

#define BLOCK_SIZE_REGISTER_I_32  32
#define BLOCK_SIZE_REGISTER_J_32  32
#define BLOCK_SIZE_REGISTER_K_32  32

static void do_block_unrolled_32 (int lda, double* restrict A, double* restrict B, double* restrict C)
{
	// transpose matrix B
	for (int k = 0; k < BLOCK_SIZE_REGISTER_K_32; k++)
	{
		for (int j = 0; j < BLOCK_SIZE_REGISTER_J_32; j++)
		{
			B_transposed[BLOCK_SIZE_REGISTER_K_32*j+k] = B[k*lda+j];
		}
	}

#ifdef BUFFER_A
	// buffer matrix A
	for (int i = 0; i < BLOCK_SIZE_REGISTER_I_32; i++) {
		memcpy(A_buffered + i*BLOCK_SIZE_REGISTER_K_32, A + i*lda, BLOCK_SIZE_REGISTER_K_32 * sizeof(double)); 
	}
#endif

	for (int i = 0; i < BLOCK_SIZE_REGISTER_I_32; ++i)
    /* For each column j of B */
		for (int j = 0; j < BLOCK_SIZE_REGISTER_J_32; ++j)
    {
      /* Compute C(i,j) */
      double cij = C[i*lda+j];

			for (int k = 0; k < BLOCK_SIZE_REGISTER_K_32; k += 8) {
				int index_A = i*BLOCK_SIZE_REGISTER_K_32 + k;
				int index_B = j*BLOCK_SIZE_REGISTER_K_32 + k;
			
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
				
				cij += _mm_cvtsd_f64(c3) + _mm_cvtsd_f64(_mm_unpackhi_pd(c3, c3));
			}

      C[i*lda+j] = cij;
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* restrict A, double* restrict B, double* restrict C)
{
	int fringe_start_i = lda / BLOCK_SIZE_REGISTER_I_32 * BLOCK_SIZE_REGISTER_I_32;
	int fringe_start_j = lda / BLOCK_SIZE_REGISTER_J_32 * BLOCK_SIZE_REGISTER_J_32;
	int fringe_start_k = lda / BLOCK_SIZE_REGISTER_K_32 * BLOCK_SIZE_REGISTER_K_32;

  /* For each block-row of A */ 
  for (int i = 0; i < fringe_start_i; i += BLOCK_SIZE_REGISTER_I_32)
    /* For each block-column of B */
    for (int j = 0; j < fringe_start_j; j += BLOCK_SIZE_REGISTER_J_32)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < fringe_start_k; k += BLOCK_SIZE_REGISTER_K_32)
      {
				do_block_unrolled_32(lda, A + i*lda + k, B + k*lda + j, C + i*lda + j);
				//do_block_not_unrolled(i, j, k, BLOCK_SIZE_REGISTER_I, BLOCK_SIZE_REGISTER_J, BLOCK_SIZE_REGISTER_K);
				//do_block_not_unrolled(i, j, k, min(lda-i, BLOCK_SIZE_REGISTER), min(lda-j, BLOCK_SIZE_REGISTER), min(lda-k, BLOCK_SIZE_REGISTER));

      }

	/* For the fringe cases */
	// {i}
	{
		int i = fringe_start_i;
		for (int j = 0; j < fringe_start_j; j += BLOCK_SIZE_REGISTER_J_32)
			for (int k = 0; k < fringe_start_k; k += BLOCK_SIZE_REGISTER_K_32) 
			{
				do_block_not_unrolled(i, j, k, lda - i, BLOCK_SIZE_REGISTER_J_32, BLOCK_SIZE_REGISTER_K_32);
			}
	}

	// {j}
  {
    int j = fringe_start_j;
    for (int i = 0; i < fringe_start_i; i += BLOCK_SIZE_REGISTER_I_32)
      for (int k = 0; k < fringe_start_k; k += BLOCK_SIZE_REGISTER_K_32)
      {
				do_block_not_unrolled(i, j, k, BLOCK_SIZE_REGISTER_I_32, lda - j, BLOCK_SIZE_REGISTER_K_32);
      }
  }

	// {k}
  {
    int k = fringe_start_k;
    for (int i = 0; i < fringe_start_i; i += BLOCK_SIZE_REGISTER_I_32)
      for (int j = 0; j < fringe_start_j; j += BLOCK_SIZE_REGISTER_J_32)
      {
				do_block_not_unrolled(i, j, k, BLOCK_SIZE_REGISTER_I_32, BLOCK_SIZE_REGISTER_J_32, lda - k);
      }
  }

	// {i, j}
	{
		int i = fringe_start_i;
		int j = fringe_start_j;
		for (int k = 0; k < fringe_start_k; k += BLOCK_SIZE_REGISTER_K_32) 
		{
			do_block_not_unrolled(i, j, k, lda - i, lda - j, BLOCK_SIZE_REGISTER_K_32);
		}
	}

	// {j, k}
  {
    int j = fringe_start_j;
    int k = fringe_start_k;
    for (int i = 0; i < fringe_start_i; i += BLOCK_SIZE_REGISTER_I_32) 
    {
      do_block_not_unrolled(i, j, k, BLOCK_SIZE_REGISTER_I_32, lda - j, lda - k);
    }
  }

	// {i, k}
  {
    int i = fringe_start_i;
    int k = fringe_start_k;
    for (int j = 0; j < fringe_start_j; j += BLOCK_SIZE_REGISTER_J_32)
    {
      do_block_not_unrolled(i, j, k, lda - i, BLOCK_SIZE_REGISTER_J_32, lda - k);
    }
  }

	// {i, j, k}
	{
		int i = fringe_start_i;
		int j = fringe_start_j;
		int k = fringe_start_k;

		do_block_not_unrolled(i, j, k, lda - i, lda - j, lda - k);
	}

	//free(B_transposed);
	//free(A_buffered);
}
