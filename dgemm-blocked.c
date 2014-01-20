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

static void do_block_unrolled (int lda, int BLOCK_SIZE_REGISTER_I, int BLOCK_SIZE_REGISTER_J, int BLOCK_SIZE_REGISTER_K, double* restrict A, double* restrict B, double* restrict C)
{
	// transpose matrix B
	for (int k = 0; k < BLOCK_SIZE_REGISTER_K; k++)
	{
		for (int j = 0; j < BLOCK_SIZE_REGISTER_J; j++)
		{
			B_transposed[BLOCK_SIZE_REGISTER_K*j+k] = B[k*lda+j];
		}
	}

#ifdef BUFFER_A
	// buffer matrix A
	for (int i = 0; i < BLOCK_SIZE_REGISTER_I; i++) {
//		for (int k = 0; k < BLOCK_SIZE_REGISTER_K; k++) {
//			A_buffered[i*BLOCK_SIZE_REGISTER_K + k] = A[i*lda + k];
//		}
		memcpy(A_buffered + i*BLOCK_SIZE_REGISTER_K, A + i*lda, BLOCK_SIZE_REGISTER_K * sizeof(double)); 
	}
#endif

	double c_d_1;
	double c_d_2;
	
  /* For each row i of A */
#ifdef INNER_DOBLOCK_LOOP
  for (int i = 0; i < BLOCK_SIZE_REGISTER_I; i += MU)
#endif
#ifndef INNER_DOBLOCK_LOOP
	for (int i = 0; i < BLOCK_SIZE_REGISTER_I; ++i)
#endif
    /* For each column j of B */
#ifdef INNER_DOBLOCK_LOOP
    for (int j = 0; j < BLOCK_SIZE_REGISTER_J; j += NU)
#endif
#ifndef INNER_DOBLOCK_LOOP
		for (int j = 0; j < BLOCK_SIZE_REGISTER_J; ++j)
#endif
    {
      /* Compute C(i,j) */
#ifndef INNER_DOBLOCK_LOOP
      double cij = C[i*lda+j];
#endif

#ifdef INNER_DOBLOCK_LOOP
      for (int k = 0; k < BLOCK_SIZE_REGISTER_K; k += KU) {
#endif
#ifndef INNER_DOBLOCK_LOOP
			for (int k = 0; k < BLOCK_SIZE_REGISTER_K; k += 8) {
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
				int index_A = i*BLOCK_SIZE_REGISTER_K + k;
				int index_B = j*BLOCK_SIZE_REGISTER_K + k;
			
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
void square_dgemm (int lda, double* restrict A, double* restrict B, double* restrict C)
{
//		BLOCK_SIZE_K = min(lda/8*8, 128);
//		BLOCK_SIZE_I = BLOCK_SIZE_J =  (1024 / 2 / BLOCK_SIZE_K) / 8 * 8;

	BLOCK_SIZE_I = BLOCK_SIZE_J = BLOCK_SIZE_K = 32;

//	printf("  BLOCK_SIZE_I: %i, BLOCK_SIZE_J: %i, BLOCK_SIZE_K: %i\n", BLOCK_SIZE_I, BLOCK_SIZE_J, BLOCK_SIZE_K);

	// will always be BLOCK_SIZE, but the compiler is weired
	register int BLOCK_SIZE_REGISTER_I = min(lda * 100000, BLOCK_SIZE_I);
	register int BLOCK_SIZE_REGISTER_J = min(lda * 100000, BLOCK_SIZE_J);
	register int BLOCK_SIZE_REGISTER_K = min(lda * 100000, BLOCK_SIZE_K);

	int fringe_start_i = lda / BLOCK_SIZE_REGISTER_I * BLOCK_SIZE_REGISTER_I;
	int fringe_start_j = lda / BLOCK_SIZE_REGISTER_J * BLOCK_SIZE_REGISTER_J;
	int fringe_start_k = lda / BLOCK_SIZE_REGISTER_K * BLOCK_SIZE_REGISTER_K;

//	B_transposed = (double*) memalign(16, sizeof(double) * BLOCK_SIZE_REGISTER_K*BLOCK_SIZE_REGISTER_J);
//	A_buffered = (double*) memalign(16, sizeof(double) * BLOCK_SIZE_REGISTER_K*BLOCK_SIZE_REGISTER_I);

  /* For each block-row of A */ 
  for (int i = 0; i < fringe_start_i; i += BLOCK_SIZE_REGISTER_I)
    /* For each block-column of B */
    for (int j = 0; j < fringe_start_j; j += BLOCK_SIZE_REGISTER_J)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < fringe_start_k; k += BLOCK_SIZE_REGISTER_K)
      {
				do_block_unrolled(lda, BLOCK_SIZE_REGISTER_I, BLOCK_SIZE_REGISTER_J, BLOCK_SIZE_REGISTER_K, A + i*lda + k, B + k*lda + j, C + i*lda + j);
				//do_block_not_unrolled(i, j, k, BLOCK_SIZE_REGISTER_I, BLOCK_SIZE_REGISTER_J, BLOCK_SIZE_REGISTER_K);
				//do_block_not_unrolled(i, j, k, min(lda-i, BLOCK_SIZE_REGISTER), min(lda-j, BLOCK_SIZE_REGISTER), min(lda-k, BLOCK_SIZE_REGISTER));

      }

	/* For the fringe cases */
	// {i}
	{
		int i = fringe_start_i;
		for (int j = 0; j < fringe_start_j; j += BLOCK_SIZE_REGISTER_J)
			for (int k = 0; k < fringe_start_k; k += BLOCK_SIZE_REGISTER_K) 
			{
				do_block_not_unrolled(i, j, k, lda - i, BLOCK_SIZE_REGISTER_J, BLOCK_SIZE_REGISTER_K);
			}
	}

	// {j}
  {
    int j = fringe_start_j;
    for (int i = 0; i < fringe_start_i; i += BLOCK_SIZE_REGISTER_I)
      for (int k = 0; k < fringe_start_k; k += BLOCK_SIZE_REGISTER_K)
      {
				do_block_not_unrolled(i, j, k, BLOCK_SIZE_REGISTER_I, lda - j, BLOCK_SIZE_REGISTER_K);
      }
  }

	// {k}
  {
    int k = fringe_start_k;
    for (int i = 0; i < fringe_start_i; i += BLOCK_SIZE_REGISTER_I)
      for (int j = 0; j < fringe_start_j; j += BLOCK_SIZE_REGISTER_J)
      {
				do_block_not_unrolled(i, j, k, BLOCK_SIZE_REGISTER_I, BLOCK_SIZE_REGISTER_J, lda - k);
      }
  }

	// {i, j}
	{
		int i = fringe_start_i;
		int j = fringe_start_j;
		for (int k = 0; k < fringe_start_k; k += BLOCK_SIZE_REGISTER_K) 
		{
			do_block_not_unrolled(i, j, k, lda - i, lda - j, BLOCK_SIZE_REGISTER_K);
		}
	}

	// {j, k}
  {
    int j = fringe_start_j;
    int k = fringe_start_k;
    for (int i = 0; i < fringe_start_i; i += BLOCK_SIZE_REGISTER_I) 
    {
      do_block_not_unrolled(i, j, k, BLOCK_SIZE_REGISTER_I, lda - j, lda - k);
    }
  }

	// {i, k}
  {
    int i = fringe_start_i;
    int k = fringe_start_k;
    for (int j = 0; j < fringe_start_j; j += BLOCK_SIZE_REGISTER_J)
    {
      do_block_not_unrolled(i, j, k, lda - i, BLOCK_SIZE_REGISTER_J, lda - k);
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
